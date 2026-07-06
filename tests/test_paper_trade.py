"""
Tests for paper_trade.py.

The load-bearing test is signal parity: for every bar of a shared price
history, the paper loop's compute_signal() must make the same decision the
real MACrossover makes inside the real engine. If those ever diverge, paper
results stop being evidence about the walk-forward result — the entire
experiment silently becomes about a different strategy.

Network is never touched: candles are injected as DataFrames.
"""
import json

import numpy as np
import pandas as pd
import pytest

import paper_trade as pt
from engine.engine import run_backtest
from strategies.ma_crossover import MACrossover

UTC = "UTC"


def make_candles(closes, start="2024-01-01"):
    """Build a candle frame in the paper loop's expected shape (incl. close_time)."""
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="4h", tz=UTC)
    close = pd.Series(closes, dtype=float)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]).values,
            "high": (close * 1.002).values,
            "low": (close * 0.998).values,
            "close": close.values,
            "close_time": [(t + pd.Timedelta(hours=4)).value // 10**6 for t in idx],
        },
        index=idx,
    )
    return df


def trend_series(n=800, seed=7):
    """Price path that guarantees crossover EVENTS after SMA warmup.

    With slow_period=200 the SMAs are only valid from bar ~200, so the shape
    keeps a downtrend well past bar 350 (fast below slow when both become
    valid), then a strong up-leg (bullish cross ~bar 400+), then a down-leg
    (bearish cross later). The first version of this helper produced ZERO
    crossovers in valid-SMA territory and the parity test passed by comparing
    two empty lists — the precondition asserts below exist so that can never
    happen silently again."""
    rng = np.random.default_rng(seed)
    a, b = int(n * 0.45), int(n * 0.8)
    legs = np.concatenate([
        np.linspace(0.0, -0.25, a),          # long downtrend through warmup
        np.linspace(-0.25, 0.30, b - a),     # strong recovery -> bullish cross
        np.linspace(0.30, 0.05, n - b),      # rollover -> bearish cross
    ])
    noise = rng.normal(0, 0.003, n).cumsum()
    return 20000 * np.exp(legs + noise)


@pytest.fixture(autouse=True)
def isolate_files(tmp_path, monkeypatch):
    """Point state/log at a temp dir so tests never touch real experiment files."""
    monkeypatch.setattr(pt, "STATE_FILE", tmp_path / "paper_state.json")
    monkeypatch.setattr(pt, "LOG_FILE", tmp_path / "paper_log.csv")


# ── THE test: paper signals == backtest signals, bar by bar ──────────────────

def test_signal_parity_with_backtest_engine():
    """Replay a full history through BOTH paths:
      - the real engine running the real MACrossover (ground truth)
      - the paper loop's compute_signal() called on each growing prefix,
        with position/peak state threaded forward exactly as run_cycle() does
    The sequence of (bar, decision) pairs must be identical."""
    closes = trend_series(400)
    full = make_candles(closes)

    # Ground truth: engine trades (entry/exit signal bars are fill_bar - 1)
    r = run_backtest(MACrossover(**pt.PARAMS), full[["open", "high", "low", "close"]])
    engine_signals = []
    for t in r.trades:
        engine_signals.append(("buy", t.entry_bar_idx - 1))
        if t.exit_reason != "end_of_data":
            engine_signals.append(("close", t.exit_bar_idx - 1))

    # Paper path: feed growing prefixes, thread state like run_cycle does
    paper_signals = []
    in_pos, peak = False, 0.0
    warm = pt.PARAMS["slow_period"] + 1
    for k in range(warm, len(full)):
        prefix = full.iloc[: k + 1]
        sig, peak, _ = pt.compute_signal(prefix, in_pos, peak)
        if sig == "buy":
            paper_signals.append(("buy", k))
            in_pos, peak = True, float(prefix["close"].iloc[-1])
        elif sig in ("close_stop", "close_cross"):
            paper_signals.append(("close", k))
            in_pos, peak = False, 0.0

    # PRECONDITION: parity between two empty lists proves nothing.
    assert engine_signals, "test data produced no engine signals — parity test is vacuous"
    assert paper_signals == engine_signals, (
        f"paper loop diverged from engine:\n paper={paper_signals}\n engine={engine_signals}"
    )


# ── fill mechanics across cycles ─────────────────────────────────────────────

def test_buy_signal_fills_next_candle_open_not_signal_close():
    state = pt.default_state()
    df = make_candles(trend_series(300))
    signal_ct = int(df["close_time"].iloc[250])
    state["pending"] = {"action": "buy", "signal_close_time": signal_ct}

    action, fill_price, _ = pt.fill_pending(state, df)
    assert action == "FILLED_BUY"
    assert fill_price == float(df["open"].iloc[251]), \
        "must fill at the FIRST candle after the signal, at its OPEN"
    assert state["size"] > 0 and state["pending"] is None
    # fee-inclusive sizing: cash is (approximately) fully deployed, never negative
    assert -1e-6 <= state["cash"] < 1.0


def test_pending_waits_when_no_newer_candle_exists():
    state = pt.default_state()
    df = make_candles(trend_series(300))
    state["pending"] = {"action": "buy",
                        "signal_close_time": int(df["close_time"].iloc[-1])}
    action, _, _ = pt.fill_pending(state, df)
    assert action == "wait"
    assert state["pending"] is not None, "pending order must survive until fillable"


def test_close_fill_round_trip_pnl_matches_hand_math():
    state = pt.default_state()
    df = make_candles([100.0] * 250 + [110.0, 111.0, 112.0])
    state["pending"] = {"action": "buy", "signal_close_time": int(df["close_time"].iloc[-3])}
    pt.fill_pending(state, df)                       # buys at open of bar -2
    entry = state["entry_price"]
    size = state["size"]
    state["pending"] = {"action": "close", "signal_close_time": int(df["close_time"].iloc[-2])}
    action, exit_price, _ = pt.fill_pending(state, df)
    assert action == "FILLED_CLOSE"
    expected_cash = size * exit_price * (1 - pt.FEE_RATE)
    assert state["cash"] == pytest.approx(expected_cash, rel=1e-12)
    assert state["size"] == 0.0


def test_stale_orders_dropped_not_double_filled():
    state = pt.default_state()
    df = make_candles(trend_series(300))
    state["size"] = 0.5   # already long
    state["pending"] = {"action": "buy", "signal_close_time": int(df["close_time"].iloc[200])}
    action, _, _ = pt.fill_pending(state, df)
    assert action == "skip" and state["pending"] is None
    state2 = pt.default_state()  # already flat
    state2["pending"] = {"action": "close", "signal_close_time": int(df["close_time"].iloc[200])}
    action2, _, _ = pt.fill_pending(state2, df)
    assert action2 == "skip" and state2["pending"] is None


# ── state persistence / restart survival ─────────────────────────────────────

def test_state_survives_save_load_round_trip():
    s = pt.default_state()
    s["cash"] = 1234.56
    s["size"] = 0.789
    s["pending"] = {"action": "close", "signal_close_time": 1700000000000}
    pt.save_state(s)
    loaded = pt.load_state()
    assert loaded == s


def test_load_state_tolerates_missing_new_fields():
    """A state file written by an older version must not crash a newer one."""
    with open(pt.STATE_FILE, "w") as f:
        json.dump({"cash": 5000.0, "size": 0.0}, f)
    s = pt.load_state()
    assert s["cash"] == 5000.0
    assert "pending" in s and "last_seen_close_time" in s


# ── full-cycle behavior with mocked network ──────────────────────────────────

def test_cycle_is_idempotent_per_candle(monkeypatch):
    """Running the cycle 3x on the SAME candle set must not re-signal or
    change state after the first run — the last_seen_close_time guard."""
    df = make_candles(trend_series(400))
    monkeypatch.setattr(pt, "fetch_closed_candles", lambda: df)
    pt.run_cycle()
    state1 = pt.load_state()
    pt.run_cycle()
    pt.run_cycle()
    state3 = pt.load_state()
    assert state1 == state3, "same candle processed twice changed state"


def test_signal_then_fill_across_two_cycles(monkeypatch):
    """End-to-end: cycle 1 sees a fresh bullish crossover -> pending buy.
    Cycle 2 sees one more candle -> fills at its open. Equity coherent."""
    closes = trend_series(400)
    full = make_candles(closes)

    # Find a bullish crossover bar using the strategy's own SMAs
    s = MACrossover(**pt.PARAMS)
    s.init(full[["open", "high", "low", "close"]])
    cross = None
    for i in range(1, len(full)):
        if (s.sma_fast.iloc[i] > s.sma_slow.iloc[i]
                and s.sma_fast.iloc[i - 1] <= s.sma_slow.iloc[i - 1]):
            cross = i
            break
    assert cross is not None and cross + 1 < len(full), "test data must contain a crossover"

    monkeypatch.setattr(pt, "fetch_closed_candles", lambda: full.iloc[: cross + 1])
    pt.run_cycle()
    st = pt.load_state()
    assert st["pending"] is not None and st["pending"]["action"] == "buy"

    monkeypatch.setattr(pt, "fetch_closed_candles", lambda: full.iloc[: cross + 2])
    pt.run_cycle()
    st = pt.load_state()
    assert st["pending"] is None
    assert st["size"] > 0
    assert st["entry_price"] == pytest.approx(float(full["open"].iloc[cross + 1]))


def test_error_is_logged_not_fatal(monkeypatch):
    """A fetch failure must produce an ERROR row and exit code 0 (so Task
    Scheduler keeps the schedule alive), never an unhandled crash."""
    def boom():
        raise RuntimeError("binance down")
    monkeypatch.setattr(pt, "fetch_closed_candles", boom)
    assert pt.main() == 0
    log = pd.read_csv(pt.LOG_FILE)
    assert (log["action"] == "ERROR").any()
    assert "binance down" in str(log["note"].iloc[-1])
