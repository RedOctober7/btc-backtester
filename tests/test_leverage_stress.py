"""
Adversarial stress tests for leverage/liquidation/funding.

These bars are hand-built to hit the exact scenarios random-walk data never
produces: crash gaps through the liquidation level, flash wicks that touch it
and recover, liquidation on the entry bar itself, stop-vs-liquidation
collisions, and funding erosion pulling the trigger closer. Every resolution
must be the PESSIMISTIC one — the engine's design rule.
"""
import numpy as np
import pandas as pd
import pytest

from engine.broker import Broker
from engine.engine import run_backtest
from strategies.base import Strategy
from strategies.ma_crossover_ls import MACrossoverLS

UTC = "UTC"


def bars(rows, start="2022-01-01", freq="4h"):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz=UTC)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    return df


class EnterLongBar0(Strategy):
    """Queue a long on bar 0; hold forever. Isolates broker/engine mechanics."""
    def init(self, data): pass
    def on_bar(self, ctx):
        if ctx.current_idx == 0 and not ctx.position.has_position:
            ctx.buy()


class EnterShortBar0(Strategy):
    def init(self, data): pass
    def on_bar(self, ctx):
        if ctx.current_idx == 0 and not ctx.position.has_position:
            ctx.sell_short()


# ── 1. Gap-through liquidation: fill must be the OPEN, not the trigger ───────

def test_gap_down_liquidation_fills_at_open_not_trigger():
    """5x long entered ~20k -> liq ~16.1k. Next bar OPENS at 14k (crash gap).
    A trigger-price fill would understate the loss; the fill must be 14k."""
    df = bars([
        (20000, 20100, 19900, 20000),   # bar 0: signal
        (20000, 20050, 19950, 20000),   # bar 1: fill long at open 20000
        (14000, 14500, 13500, 14000),   # bar 2: opens far below liq -> gap fill
    ])
    r = run_backtest(EnterLongBar0(), df, leverage=5.0)
    liq_trades = [t for t in r.trades if t.exit_reason == "liquidation"]
    assert len(liq_trades) == 1
    assert liq_trades[0].exit_price == pytest.approx(14000, rel=1e-9), \
        "gap liquidation must fill at the open, not the (better) trigger price"
    assert r.equity.min() >= 0


def test_gap_up_liquidation_short_fills_at_open():
    """Mirror case: 5x short, bar opens far ABOVE the trigger."""
    df = bars([
        (20000, 20100, 19900, 20000),
        (20000, 20050, 19950, 20000),   # fill short at 20000
        (26000, 26500, 25800, 26000),   # opens way above short liq (~23.9k)
    ])
    r = run_backtest(EnterShortBar0(), df, leverage=5.0)
    liq = [t for t in r.trades if t.exit_reason == "liquidation"]
    assert len(liq) == 1
    assert liq[0].exit_price == pytest.approx(26000, rel=1e-9)
    assert r.equity.min() >= 0


# ── 2. Intrabar touch: open on the safe side, wick through -> trigger fill ───

def test_flash_wick_liquidates_at_trigger():
    """Bar opens ABOVE the long trigger but its LOW wicks through it, closing
    back high. Close-only checking would miss this entirely; the engine must
    liquidate at the trigger price."""
    df = bars([
        (20000, 20100, 19900, 20000),
        (20000, 20050, 19950, 20000),   # fill 5x long at 20000, liq ~16.1k
        (19000, 19500, 15500, 19400),   # wick down through liq, recovers
    ])
    r = run_backtest(EnterLongBar0(), df, leverage=5.0)
    liq = [t for t in r.trades if t.exit_reason == "liquidation"]
    assert len(liq) == 1, "flash wick through the trigger must liquidate"
    b = Broker(initial_capital=10_000, fee_rate=0.001, leverage=5.0)
    b.fill_buy(20000, df.index[1], 1)
    expected_trigger = b.liquidation_price
    assert liq[0].exit_price == pytest.approx(expected_trigger, rel=1e-6), \
        "intrabar (non-gap) liquidation fills at the trigger price"


def test_wick_that_does_not_reach_trigger_survives():
    df = bars([
        (20000, 20100, 19900, 20000),
        (20000, 20050, 19950, 20000),   # 5x long, liq ~16.1k
        (19000, 19500, 17000, 19400),   # low 17000 stays above trigger
    ])
    r = run_backtest(EnterLongBar0(), df, leverage=5.0)
    assert not any(t.exit_reason == "liquidation" for t in r.trades)


# ── 3. Same-bar entry + liquidation ───────────────────────────────────────────

def test_liquidation_on_entry_bar_itself():
    """Order fills at bar t's open and the SAME bar crashes through the fresh
    liquidation level. The engine must catch it that bar, not let the position
    coast to the next one."""
    df = bars([
        (20000, 20100, 19900, 20000),   # signal bar
        (20000, 20050, 15000, 15200),   # fills at 20000, then crashes through liq intrabar
        (15200, 15300, 15100, 15200),
    ])
    r = run_backtest(EnterLongBar0(), df, leverage=5.0)
    liq = [t for t in r.trades if t.exit_reason == "liquidation"]
    assert len(liq) == 1
    assert liq[0].entry_bar_idx == liq[0].exit_bar_idx == 1, \
        "entry-bar liquidation must resolve on the entry bar"


# ── 4. Stop vs liquidation collision ─────────────────────────────────────────

def test_liquidation_wins_over_stop_on_same_bar():
    """A bar that violates BOTH the strategy's stop (close-evaluated, exits next
    bar) and the liquidation trigger (intrabar). Pessimistic resolution:
    liquidation, at the liquidation fill, tagged 'liquidation'."""
    df = bars([
        (20000, 20100, 19900, 20100),   # crossover-ish setup bar (signal below)
        (20000, 20050, 19950, 20000),   # fill long 5x at 20000
        (19000, 19100, 15800, 16000),   # breaches liq (~16.1k) intrabar AND closes -20% (stop)
        (16000, 16100, 15900, 16000),
    ])
    r = run_backtest(EnterLongBar0(), df, leverage=5.0)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.exit_reason == "liquidation", \
        "when stop and liquidation collide on one bar, liquidation (pessimistic) wins"


# ── 5. Funding ────────────────────────────────────────────────────────────────

def test_funding_erodes_long_margin_and_pnl():
    """Flat price, positive funding: a leveraged long bleeds funding while price
    goes nowhere. PnL must be negative by ~(funding + fees), and funding_paid
    must be recorded on the trade."""
    n = 60  # 10 days of 4h bars, funding applies on every 8h-aligned bar
    df = bars([(20000, 20010, 19990, 20000)] * n)
    r = run_backtest(EnterLongBar0(), df, leverage=3.0, funding_rate_8h=0.0005)
    t = r.trades[0]
    assert t.funding_paid > 0, "long must PAY positive funding"
    assert t.pnl < -t.fees_paid * 0.5, "funding must show up as real PnL erosion"


def test_funding_credits_short():
    n = 60
    df = bars([(20000, 20010, 19990, 20000)] * n)
    r = run_backtest(EnterShortBar0(), df, leverage=3.0, funding_rate_8h=0.0005)
    t = r.trades[0]
    assert t.funding_paid < 0, "short RECEIVES positive funding (recorded as negative paid)"


def test_funding_pulls_liquidation_closer():
    """Sustained funding erodes margin, so the liquidation trigger must creep
    toward the mark. A long that survives price-wise can still get liquidated
    by funding erosion — verify the trigger actually moves."""
    b = Broker(initial_capital=10_000, fee_rate=0.001, leverage=10.0, funding_rate_8h=0.001)
    b.fill_buy(20000, pd.Timestamp("2022-01-01", tz=UTC), 0)
    trig_before = b.liquidation_price
    for i in range(30):
        b.apply_funding(20000, pd.Timestamp("2022-01-01", tz=UTC) + pd.Timedelta(hours=8 * i))
    trig_after = b.liquidation_price
    assert trig_after > trig_before, "funding erosion must pull the long trigger UP toward the mark"


def test_zero_funding_and_lev1_is_exact_v1():
    """The global invariant: defaults reproduce v1 bit-for-bit."""
    b = Broker(initial_capital=10_000, fee_rate=0.001)  # all defaults
    b.fill_buy(20000, pd.Timestamp("2022-01-01", tz=UTC), 0)
    size_v1 = 10000 / (20000 * 1.001)
    assert b.size == pytest.approx(size_v1, rel=1e-12)
    b.fill_close(25000, pd.Timestamp("2022-02-01", tz=UTC), "signal", 10)
    fee_in = size_v1 * 20000 * 0.001
    fee_out = size_v1 * 25000 * 0.001
    pnl_v1 = (size_v1 * 25000 - fee_out) - (size_v1 * 20000 + fee_in)
    assert b.trades[0].pnl == pytest.approx(pnl_v1, rel=1e-12)


# ── 6. Liquidation penalty & extreme leverage sweep ───────────────────────────

def test_liquidation_penalty_worsens_fill():
    df = bars([
        (20000, 20100, 19900, 20000),
        (20000, 20050, 19950, 20000),
        (19000, 19500, 15500, 19400),   # wick through 5x trigger
    ])
    r_clean = run_backtest(EnterLongBar0(), df, leverage=5.0)
    r_pen = run_backtest(EnterLongBar0(), df, leverage=5.0, liquidation_penalty_bps=50)
    fill_clean = [t for t in r_clean.trades if t.exit_reason == "liquidation"][0].exit_price
    fill_pen = [t for t in r_pen.trades if t.exit_reason == "liquidation"][0].exit_price
    assert fill_pen < fill_clean, "penalty must make a forced long-liquidation fill WORSE"


def test_no_negative_equity_across_extreme_settings():
    rng = np.random.default_rng(7)
    n = 2000
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz=UTC)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))   # violent, no drift
    o = close * (1 + rng.normal(0, 0.002, n))
    hi = np.maximum(o, close) * 1.01
    lo = np.minimum(o, close) * 0.99
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close}, index=idx)
    for lev in [2, 5, 10, 25, 50]:
        for fr in [0.0, 0.0005]:
            r = run_backtest(MACrossoverLS(10, 50, 0.08), df,
                             leverage=float(lev), funding_rate_8h=fr,
                             liquidation_penalty_bps=25)
            assert r.equity.min() >= 0, f"negative equity at lev={lev}, funding={fr}"
