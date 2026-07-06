"""
paper_trade.py — forward paper-trading loop for btc-backtester.

One invocation = one cycle: fetch closed 4h candles from Binance's public API,
run the SAME MACrossover the walk-forward validated, reconcile desired vs
actual position, log any paper fill, persist state, exit. Schedule it every
4 hours (Windows Task Scheduler); it is deliberately stateless between runs.

Design rules (each exists because its absence is a classic live-trading bug):

  CLOSED CANDLES ONLY. Binance returns the still-forming candle last; its
  close mutates until the period ends. Acting on it makes results
  irreproducible. We drop it and evaluate only sealed history.

  FILL AT NEXT OPEN, like the backtest. The engine's semantics are signal on
  bar t's close -> fill at bar t+1's open. Live, that means: a signal on the
  most recent CLOSED candle is recorded as PENDING, and filled on the next
  cycle at the first new candle's open. Paper results stay directly
  comparable to the walk-forward numbers. The one divergence to expect:
  pandas SMA vs incremental live values can differ by float dust on regime
  boundaries; entries may occasionally differ by one bar. Logged, not hidden.

  RESTART-PROOF. All position state lives in paper_state.json, written
  atomically (temp file + os.replace) so a crash mid-write can't corrupt it.
  A reboot costs at most one cycle.

  SELF-HEALING. Every cycle refetches the last ~300 candles and recomputes
  indicators from scratch — a missed cycle (PC asleep, API hiccup) needs no
  gap repair; the next cycle simply sees more history. If multiple candles
  arrived while we were away, a pending fill uses the open of the FIRST
  candle after its signal (recorded by signal close_time), not whatever is
  newest — again matching backtest fills.

  NO SILENT DEATH. Any exception is logged to paper_log.csv with action
  ERROR and the loop exits 0 so Task Scheduler keeps scheduling. The
  experiment survives; the row tells you what happened.

Strategy under test (walk-forward winner, long-only):
  MACrossover(fast_period=20, slow_period=200, stop_pct=0.08)
  53% WFE / OOS Sharpe 0.81 on dev data. Paper trading exists to find out
  whether that edge survives live-forward data. Do not tune it mid-flight:
  every parameter change resets the experiment clock to zero.

Files (created next to this script):
  paper_state.json  — position + pending-order state (the source of truth)
  paper_log.csv     — one row per cycle; the experiment's result
Read the CSV in a few weeks. That's the whole interface.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Reuse the exact validated strategy code — no reimplementation drift.
from strategies.ma_crossover import MACrossover

# ── experiment configuration (change = new experiment, restart the clock) ───
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
PARAMS = {"fast_period": 20, "slow_period": 200, "stop_pct": 0.08}
FETCH_BARS = 300            # > slow_period + cushion
INITIAL_CAPITAL = 10_000.0
FEE_RATE = 0.001            # same as backtest

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / "paper_state.json"
LOG_FILE = HERE / "paper_log.csv"

BINANCE_URL = (
    "https://api.binance.com/api/v3/klines"
    f"?symbol={SYMBOL}&interval={INTERVAL}&limit={FETCH_BARS}"
)

LOG_COLUMNS = [
    "run_at_utc", "candle_close_utc", "close_price",
    "position", "action", "fill_price", "size", "cash", "equity", "note",
]


# ── data ─────────────────────────────────────────────────────────────────────

def fetch_closed_candles() -> pd.DataFrame:
    """Fetch klines and DROP the still-forming last candle.

    Binance kline fields: [open_time, open, high, low, close, volume,
    close_time(ms), ...]. A candle is closed iff its close_time is in the
    past. The API returns the live candle as the final row; slicing it off by
    timestamp (not blindly [-1]) also behaves correctly in the rare case the
    fetch lands exactly on a boundary.
    """
    req = urllib.request.Request(BINANCE_URL, headers={"User-Agent": "paper-trader"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read())

    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    rows = [r for r in raw if r[6] < now_ms]  # close_time strictly past
    if not rows:
        raise RuntimeError("no closed candles returned")

    df = pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "close_time": [int(r[6]) for r in rows],
        },
        index=pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True),
    )
    return df


# ── state ────────────────────────────────────────────────────────────────────

def default_state() -> dict:
    return {
        "cash": INITIAL_CAPITAL,
        "size": 0.0,               # BTC held (long-only: 0 or positive)
        "entry_price": 0.0,
        "peak_close": 0.0,         # trailing anchor for the drawdown stop
        "pending": None,           # {"action": "buy"|"close", "signal_close_time": ms}
        "last_seen_close_time": 0, # ms; idempotency guard
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            s = json.load(f)
        base = default_state()
        base.update(s)             # forward-compatible if fields are added
        return base
    return default_state()


def save_state(state: dict) -> None:
    """Atomic write: crash mid-save leaves the old state intact, never half a file."""
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def log_row(**kw) -> None:
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({c: kw.get(c, "") for c in LOG_COLUMNS})


# ── signal (identical math to the backtest strategy) ─────────────────────────

def compute_signal(df: pd.DataFrame, in_position: bool, peak_close: float):
    """Evaluate MACrossover's decision on the LAST CLOSED candle.

    Returns (action, new_peak_close, note):
      action in {None, "buy", "close_stop", "close_cross"}.
    Uses the strategy class's own precomputed SMAs (init()) so the arithmetic
    is literally the code the walk-forward validated, then applies the same
    decision rules as MACrossover.on_bar() to the final row.
    """
    strat = MACrossover(**PARAMS)
    strat.init(df)
    i = len(df) - 1
    fast, slow = strat.sma_fast.iloc[i], strat.sma_slow.iloc[i]
    prev_fast, prev_slow = strat.sma_fast.iloc[i - 1], strat.sma_slow.iloc[i - 1]
    if pd.isna(fast) or pd.isna(slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
        return None, peak_close, "warmup (SMA not ready)"

    close = float(df["close"].iloc[i])

    if in_position:
        peak_close = max(peak_close, close)
        drawdown = (peak_close - close) / peak_close
        if drawdown >= PARAMS["stop_pct"]:
            return "close_stop", peak_close, f"drawdown {drawdown:.2%} >= stop"
        if fast < slow and prev_fast >= prev_slow:
            return "close_cross", peak_close, "bearish crossover"
        return None, peak_close, "holding"
    else:
        if fast > slow and prev_fast <= prev_slow:
            return "buy", close, "bullish crossover"
        return None, peak_close, "flat"


# ── fills (backtest semantics: next candle's open, fee-inclusive sizing) ─────

def fill_pending(state: dict, df: pd.DataFrame) -> tuple[str, float, str]:
    """Fill a pending order at the open of the FIRST candle strictly after the
    signal candle. Returns (action_label, fill_price, note) or a wait/stale
    marker. Uses the identical sizing/fee formulas as engine.broker at
    leverage=1, so paper PnL is comparable to backtest PnL."""
    pending = state["pending"]
    after = df[df["close_time"] > pending["signal_close_time"]]
    if after.empty:
        return "wait", 0.0, "no candle after signal yet"

    fill_price = float(after["open"].iloc[0])
    if pending["action"] == "buy":
        if state["size"] > 0:
            state["pending"] = None
            return "skip", 0.0, "already long; stale buy dropped"
        target = state["cash"]
        size = target / (fill_price * (1 + FEE_RATE))   # fee-inclusive, v1 formula
        state["cash"] -= size * fill_price * (1 + FEE_RATE)
        state["size"] = size
        state["entry_price"] = fill_price
        state["peak_close"] = fill_price
        state["pending"] = None
        return "FILLED_BUY", fill_price, f"size {size:.6f} BTC"
    else:  # close
        if state["size"] == 0:
            state["pending"] = None
            return "skip", 0.0, "already flat; stale close dropped"
        proceeds = state["size"] * fill_price
        state["cash"] += proceeds - proceeds * FEE_RATE
        pnl_note = f"exit {fill_price:.2f} vs entry {state['entry_price']:.2f}"
        state["size"] = 0.0
        state["entry_price"] = 0.0
        state["peak_close"] = 0.0
        state["pending"] = None
        return "FILLED_CLOSE", fill_price, pnl_note


# ── one cycle ────────────────────────────────────────────────────────────────

def run_cycle() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = load_state()

    df = fetch_closed_candles()
    last = df.iloc[-1]
    last_close_time = int(last["close_time"])
    close_price = float(last["close"])

    # Step 1: fill any pending order (uses first candle AFTER its signal)
    action, fill_price, note = "", 0.0, ""
    if state["pending"] is not None:
        action, fill_price, note = fill_pending(state, df)

    # Step 2: evaluate signal on the newest closed candle — once (idempotent)
    if last_close_time > state["last_seen_close_time"]:
        in_pos = state["size"] > 0
        sig, new_peak, sig_note = compute_signal(df, in_pos, state["peak_close"])
        state["peak_close"] = new_peak
        state["last_seen_close_time"] = last_close_time
        if sig == "buy" and state["pending"] is None and not in_pos:
            state["pending"] = {"action": "buy", "signal_close_time": last_close_time}
            note = (note + " | " if note else "") + f"SIGNAL buy ({sig_note}) -> fills next open"
        elif sig in ("close_stop", "close_cross") and state["pending"] is None and in_pos:
            state["pending"] = {"action": "close", "signal_close_time": last_close_time}
            note = (note + " | " if note else "") + f"SIGNAL {sig} ({sig_note}) -> fills next open"
        else:
            note = (note + " | " if note else "") + sig_note
    else:
        note = (note + " | " if note else "") + "candle already processed"

    equity = state["cash"] + state["size"] * close_price
    position = "LONG" if state["size"] > 0 else "FLAT"
    if state["pending"]:
        position += f" (pending {state['pending']['action']})"

    save_state(state)
    log_row(
        run_at_utc=now,
        candle_close_utc=pd.to_datetime(last_close_time, unit="ms", utc=True).isoformat(),
        close_price=f"{close_price:.2f}",
        position=position,
        action=action or "none",
        fill_price=f"{fill_price:.2f}" if fill_price else "",
        size=f"{state['size']:.6f}",
        cash=f"{state['cash']:.2f}",
        equity=f"{equity:.2f}",
        note=note,
    )
    print(f"[{now}] {position} | close {close_price:.2f} | equity {equity:.2f} | {note}")


def main() -> int:
    """One cycle with total failure containment. Always returns 0: an API
    hiccup or bug must be a logged CSV row, never a Task Scheduler failure
    state that silently stops future runs."""
    try:
        run_cycle()
    except Exception as e:  # noqa: BLE001 — survive anything, log it, keep the schedule
        log_row(
            run_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            action="ERROR",
            note=f"{type(e).__name__}: {e}",
        )
        print(f"ERROR logged: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
