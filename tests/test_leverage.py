"""
Tests for the v2 broker: short selling, leverage, and liquidation.

The load-bearing test is `test_leverage_1_long_matches_v1`: it pins the exact
numbers the original spot-only broker produced, so any future change that would
alter existing backtest/README results fails loudly here.
"""
import numpy as np
import pandas as pd
import pytest

from engine.broker import Broker
from engine.engine import run_backtest
from strategies.ma_crossover import MACrossover
from strategies.ma_crossover_ls import MACrossoverLS


TS = pd.Timestamp("2022-01-01", tz="UTC")
TS2 = pd.Timestamp("2022-02-01", tz="UTC")


def _synth(seed=4, n=3000, drift=0.0003, vol=0.015):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    close = 20000 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = close * (1 + rng.normal(0, 0.001, n))
    hi = np.maximum(o, close) * 1.003
    lo = np.minimum(o, close) * 0.997
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close}, index=idx)


# ── v1 equivalence ──────────────────────────────────────────────────────────

def test_leverage_1_long_matches_v1():
    """At leverage=1, a long round-trip reproduces the exact v1 formulas."""
    b = Broker(initial_capital=10_000, fee_rate=0.001, leverage=1.0)
    b.fill_buy(20000, TS, 0)
    size_v1 = 10000 / (20000 * 1.001)
    assert b.size == pytest.approx(size_v1, rel=1e-12)
    assert b.cash == pytest.approx(0.0, abs=1e-9)
    b.fill_close(25000, TS2, "signal", 100)
    fee_in = size_v1 * 20000 * 0.001
    fee_out = size_v1 * 25000 * 0.001
    pnl_v1 = (size_v1 * 25000 - fee_out) - (size_v1 * 20000 + fee_in)
    assert b.trades[0].pnl == pytest.approx(pnl_v1, rel=1e-12)
    assert b.trades[0].direction == "long"


def test_existing_strategy_unchanged():
    """The long-only MACrossover through the new engine stays long-only, no liqs."""
    data = _synth()
    r = run_backtest(MACrossover(20, 100, 0.08), data)  # default leverage=1
    assert all(t.direction == "long" for t in r.trades)
    assert all(t.exit_reason != "liquidation" for t in r.trades)
    assert r.equity.iloc[-1] > 0


# ── shorting ────────────────────────────────────────────────────────────────

def test_short_profits_when_price_falls():
    b = Broker(initial_capital=10_000, fee_rate=0.001, leverage=1.0)
    b.fill_short(20000, TS, 0)
    assert b.direction == "short"
    b.fill_close(18000, TS2, "signal", 50)          # price fell 10%
    assert b.trades[0].pnl > 0
    assert b.trades[0].direction == "short"


def test_short_loses_when_price_rises():
    b = Broker(initial_capital=10_000, leverage=1.0)
    b.fill_short(20000, TS, 0)
    assert b.equity(22000) < 10_000                 # +10% against a short = loss


# ── leverage & liquidation ──────────────────────────────────────────────────

def test_leverage_scales_notional_not_margin():
    b = Broker(initial_capital=10_000, fee_rate=0.001, leverage=5.0)
    b.fill_buy(20000, TS, 0)
    assert b.size * 20000 == pytest.approx(50_000, rel=1e-3)   # 5x notional
    assert b.cash == pytest.approx(0.0, abs=1e-6)              # all cash posted as margin


def test_liquidation_price_symmetry():
    long_b = Broker(initial_capital=10_000, leverage=5.0)
    long_b.fill_buy(20000, TS, 0)
    short_b = Broker(initial_capital=10_000, leverage=5.0)
    short_b.fill_short(20000, TS, 0)
    assert long_b.liquidation_price < 20000 < short_b.liquidation_price
    dn = 20000 - long_b.liquidation_price
    up = short_b.liquidation_price - 20000
    assert dn == pytest.approx(up, rel=0.05)


def test_equity_never_negative_any_leverage():
    """The core safety invariant: no leverage setting can drive equity below 0."""
    data = _synth()
    for lev in [1, 2, 3, 5, 10, 25, 50, 100]:
        r = run_backtest(MACrossoverLS(20, 100, 0.08), data, leverage=float(lev))
        assert r.equity.min() >= 0, f"negative equity at {lev}x"


def test_liquidation_triggers_at_high_leverage():
    data = _synth()
    r = run_backtest(MACrossoverLS(20, 100, 0.08), data, leverage=25.0)
    assert any(t.exit_reason == "liquidation" for t in r.trades)


def test_leverage_below_one_rejected():
    with pytest.raises(ValueError):
        Broker(leverage=0.5)
