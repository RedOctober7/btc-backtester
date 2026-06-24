"""
Metric correctness tests using hand-computed values.
Every test here works out the expected value by hand, then asserts the function
matches. This catches annualization errors and formula bugs early.
"""
import math
import numpy as np
import pandas as pd
import pytest
from metrics.metrics import (
    total_return,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    exposure,
    trade_stats,
    BARS_PER_YEAR,
)
from engine.broker import Trade


def _make_equity(values: list[float], freq: str = "4h") -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=dates, name="equity")


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------

def test_total_return_hand_computed():
    # 1000 -> 1500 = +50%
    equity = _make_equity([1000.0, 1100.0, 1200.0, 1500.0])
    assert abs(total_return(equity) - 50.0) < 1e-8


def test_total_return_negative():
    equity = _make_equity([1000.0, 500.0])
    assert abs(total_return(equity) - (-50.0)) < 1e-8


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_hand_computed():
    """
    Equity: 1000, 1100, 900, 800, 1000
    Running max: 1000, 1100, 1100, 1100, 1100
    Drawdowns:   0,    0,  -18.18%, -27.27%, -9.09%
    Max drawdown = -300/1100 = -27.2727...%
    """
    equity = _make_equity([1000.0, 1100.0, 900.0, 800.0, 1000.0])
    dd, duration = max_drawdown(equity)
    expected = -300.0 / 1100.0 * 100
    assert abs(dd - expected) < 1e-6, f"Expected {expected:.6f}%, got {dd:.6f}%"


def test_max_drawdown_no_drawdown():
    """Monotonically rising equity has 0% drawdown."""
    equity = _make_equity([1000.0, 1100.0, 1200.0, 1300.0])
    dd, _ = max_drawdown(equity)
    assert abs(dd) < 1e-8


# ---------------------------------------------------------------------------
# Sharpe annualization
# ---------------------------------------------------------------------------

def test_sharpe_uses_sqrt_2190_annualization():
    """
    Sharpe = mean(returns) / std(returns) * sqrt(BARS_PER_YEAR).
    We verify the factor is exactly sqrt(2190), not sqrt(252) or sqrt(365).
    """
    equity = _make_equity([1000.0, 1010.0, 990.0, 1020.0, 1005.0])
    ret = equity.pct_change().dropna()
    expected = (ret.mean() / ret.std()) * math.sqrt(BARS_PER_YEAR)
    assert abs(sharpe_ratio(equity) - expected) < 1e-10
    assert BARS_PER_YEAR == 2190, (
        f"BARS_PER_YEAR must be 2190 (6 bars/day × 365 days). Got {BARS_PER_YEAR}."
    )


def test_sharpe_full_series_not_only_trade_bars():
    """
    Sharpe is computed on the FULL equity curve — including flat bars where
    the strategy holds no position (return = 0). Not only bars inside a trade.
    The two approaches give materially different numbers; the full-series
    version is the honest one.
    """
    # 10 bars of flat (no position) then 4 bars of gain
    flat = [1000.0] * 10
    gains = [1000.0, 1050.0, 1100.0, 1150.0]
    equity = _make_equity(flat + gains)
    ret = equity.pct_change().dropna()
    # If Sharpe were computed only on the 4 gain-bars, it would be very high
    # (no variance from flat bars diluting the denominator)
    expected_full = (ret.mean() / ret.std()) * math.sqrt(BARS_PER_YEAR)
    assert abs(sharpe_ratio(equity) - expected_full) < 1e-10


def test_sharpe_zero_std_returns_zero():
    """A constant equity curve (zero std) returns 0 Sharpe, not division-by-zero."""
    equity = _make_equity([1000.0, 1000.0, 1000.0, 1000.0])
    assert sharpe_ratio(equity) == 0.0


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def test_exposure_hand_computed():
    # 3 bars in position out of 5 total = 60%
    in_pos = pd.Series([False, True, True, True, False])
    assert abs(exposure(in_pos) - 60.0) < 1e-8


def test_exposure_always_flat():
    in_pos = pd.Series([False, False, False])
    assert exposure(in_pos) == 0.0


# ---------------------------------------------------------------------------
# Trade stats
# ---------------------------------------------------------------------------

def _make_trade(pnl_override: float, entry_price: float = 100.0) -> Trade:
    """Build a minimal closed Trade with a specific PnL (via price manipulation)."""
    size = 10.0  # 10 BTC
    fee = 0.0
    # exit_price that yields the desired pnl: pnl = size * exit_price - size * entry_price
    exit_price = entry_price + pnl_override / size
    return Trade(
        entry_time=pd.Timestamp("2020-01-01", tz="UTC"),
        entry_price=entry_price,
        size=size,
        fee_entry=fee,
        entry_bar_idx=0,
        exit_time=pd.Timestamp("2020-01-02", tz="UTC"),
        exit_price=exit_price,
        fee_exit=fee,
        exit_bar_idx=1,
        exit_reason="signal",
    )


def test_trade_stats_win_rate():
    trades = [_make_trade(100.0), _make_trade(-50.0), _make_trade(200.0)]
    stats = trade_stats(trades)
    assert stats["num_trades"] == 3
    assert abs(stats["win_rate"] - 100.0 * 2 / 3) < 1e-8


def test_trade_stats_profit_factor():
    """profit_factor = gross_profit / gross_loss."""
    trades = [_make_trade(300.0), _make_trade(-100.0)]
    stats = trade_stats(trades)
    assert abs(stats["profit_factor"] - 3.0) < 1e-8


def test_trade_stats_no_trades():
    stats = trade_stats([])
    assert stats["num_trades"] == 0
    assert pd.isna(stats["win_rate"])
    assert pd.isna(stats["profit_factor"])


def test_trade_stats_all_wins():
    trades = [_make_trade(100.0), _make_trade(50.0)]
    stats = trade_stats(trades)
    assert stats["win_rate"] == 100.0
    # No losses -> profit_factor is infinity
    assert stats["profit_factor"] == float("inf")
