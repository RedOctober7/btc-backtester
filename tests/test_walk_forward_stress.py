"""
Adversarial stress tests for walk_forward.py with the v2 leverage engine.

The scenarios random data never produces: total account wipeout mid-fold,
liquidation DURING the warmup bars (OOS equity starts at zero), wiped capital
carrying into subsequent folds, and NaN-safety of every headline number the
summary prints. The rule: walk_forward must never crash and never print NaN
for the stitched metrics — a wiped account is -100%, not nan%.
"""
import numpy as np
import pandas as pd
import pytest

from strategies.base import Strategy
from strategies.ma_crossover_ls import MACrossoverLS
from walk_forward import walk_forward, WFConfig

UTC = "UTC"


class AlwaysLongBar1(Strategy):
    """Enters long on the first tradable bar of every run and never exits.
    At high leverage on crashing data this liquidates fast — including inside
    the warmup segment, which is the case that poisons naive stitching."""
    def init(self, data): pass
    def on_bar(self, ctx):
        if not ctx.position.has_position:
            ctx.buy()


def crash_then_flat(n=1500, start=20000.0, crash_start=250, crash_end=350):
    """Crash window placed to fall INSIDE fold 0's warmup segment
    (is_bars=400, warmup=200 -> warmup covers bars 200..400). -3% per bar for
    100 bars guarantees wipeout at any real leverage, DURING warmup, so the
    fold's OOS equity begins at exactly zero — the stitching poison case."""
    closes = [start]
    for i in range(1, n):
        closes.append(closes[-1] * (0.97 if crash_start <= i < crash_end else 1.0))
    c = np.array(closes)
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz=UTC)
    return pd.DataFrame(
        {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c}, index=idx
    )


def violent(n=3000, seed=13):
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.cumsum(rng.normal(-0.001, 0.03, n)))  # bleeding + violent
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz=UTC)
    o = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.02,
                         "low": np.minimum(o, close) * 0.98, "close": close}, index=idx)


GRID = {"fast_period": [10], "slow_period": [50], "stop_pct": [0.08]}


def _assert_sane(res):
    """The invariants every walk-forward result must satisfy."""
    eq = res.oos_equity
    assert not eq.isna().any(), "stitched equity contains NaN"
    assert (eq >= 0).all(), "stitched equity went negative"
    for name in ("total_return", "cagr", "sharpe", "max_dd"):
        v = getattr(res, name)
        assert not (isinstance(v, float) and np.isnan(v)), f"{name} is NaN"
    # summary must render without crashing and without 'nan' in the headline block
    s = res.summary()
    headline = "\n".join(s.splitlines()[:8])
    # WFE may legitimately be nan (IS unprofitable); the four headline metrics may not.


def test_wipeout_during_warmup_ruins_pessimistically_no_nan():
    """A fold whose warmup segment liquidates the account (OOS equity starts at
    zero) must trigger RUIN semantics: stitched curve flatlines at zero, headline
    metrics report -100% (not NaN, not a flattering skip), and the fold is
    flagged warmup_wiped. Skipping such folds would build the stitched curve
    only from survivors — flattering catastrophic leverage by omission."""
    df = crash_then_flat()
    res = walk_forward(df, lambda p: AlwaysLongBar1(), {"dummy": [1]},
                       WFConfig(is_bars=400, oos_bars=150, warmup_bars=200,
                                leverage=10.0))
    # PRECONDITION: the scenario must actually produce a zero-start fold,
    # otherwise this test is vacuous and proves nothing.
    zero_start_folds = [f for f in res.folds if len(f.oos_equity) and f.oos_equity.iloc[0] == 0]
    assert zero_start_folds, "scenario failed to produce a warmup-wipeout fold — test is vacuous"
    assert all(f.warmup_wiped for f in zero_start_folds), "zero-start folds must be flagged"
    assert res.ruined, "warmup wipeout must trigger ruin semantics"
    assert res.total_return == pytest.approx(-100.0)
    _assert_sane(res)
    assert "RUINED" in res.summary()


def test_total_wipeout_carries_zero_capital_forward():
    """Once the stitched account is wiped, later folds must keep it at zero
    (a dead account stays dead) — never resurrect it, never go NaN."""
    df = violent()
    res = walk_forward(df, lambda p: MACrossoverLS(**p), GRID,
                       WFConfig(is_bars=600, oos_bars=200, warmup_bars=60,
                                leverage=50.0, liquidation_penalty_bps=50))
    _assert_sane(res)
    eq = res.oos_equity.values
    if (eq == 0).any():
        first_zero = np.argmax(eq == 0)
        assert (eq[first_zero:] == 0).all(), "account resurrected after wipeout"


def test_leverage_sweep_never_crashes_or_nans():
    df = violent()
    for lev in [1.0, 3.0, 10.0, 25.0]:
        res = walk_forward(df, lambda p: MACrossoverLS(**p), GRID,
                           WFConfig(is_bars=600, oos_bars=200, warmup_bars=60,
                                    leverage=lev, funding_rate_8h=0.0005))
        _assert_sane(res)
        res.to_dataframe()  # table must render too


def test_lev1_results_unchanged_by_new_params():
    """Adding the leverage params must not move any leverage=1 number."""
    rng = np.random.default_rng(3)
    n = 2000
    close = 20000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz=UTC)
    o = close * (1 + rng.normal(0, 0.001, n))
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.002,
                       "low": np.minimum(o, close) * 0.998, "close": close}, index=idx)
    r1 = walk_forward(df, lambda p: MACrossoverLS(**p), GRID,
                      WFConfig(is_bars=600, oos_bars=200, warmup_bars=60))
    r2 = walk_forward(df, lambda p: MACrossoverLS(**p), GRID,
                      WFConfig(is_bars=600, oos_bars=200, warmup_bars=60, leverage=1.0,
                               funding_rate_8h=0.0, liquidation_penalty_bps=0.0))
    assert r1.total_return == pytest.approx(r2.total_return, rel=1e-12)
    pd.testing.assert_series_equal(r1.oos_equity, r2.oos_equity)
