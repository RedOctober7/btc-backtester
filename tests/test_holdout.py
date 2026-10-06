"""
Holdout split + one-shot holdout test.

The holdout only means something if no fold ever saw it. These tests pin the
split boundary, the warmup trimming, and the ruin-safe reporting.
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy
from strategies.ma_crossover_ls import MACrossoverLS
from walk_forward import walk_forward, WFConfig, split_holdout, holdout_test

UTC = "UTC"


def random_walk(n=2400, seed=7, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range(start, periods=n, freq="4h", tz=UTC)
    o = close * (1 + rng.normal(0, 0.001, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.005,
                         "low": np.minimum(o, close) * 0.995, "close": close}, index=idx)


class AlwaysLong(Strategy):
    def init(self, data): pass
    def on_bar(self, ctx):
        if not ctx.position.has_position:
            ctx.buy()


GRID = {"fast_period": [10, 20], "slow_period": [50], "stop_pct": [0.08]}
CFG = WFConfig(is_bars=600, oos_bars=150, warmup_bars=60)


def test_split_is_disjoint_and_ordered():
    df = random_walk()
    dev, hold = split_holdout(df, "2022-12-01")
    assert len(dev) + len(hold) == len(df)
    assert dev.index[-1] < pd.Timestamp("2022-12-01", tz=UTC) <= hold.index[0]


def test_folds_never_reach_holdout():
    df = random_walk()
    dev, hold = split_holdout(df, "2022-12-01")
    factory = lambda p: MACrossoverLS(**p)
    res = walk_forward(dev, factory, GRID, CFG)
    h = holdout_test(dev, hold, factory, res.folds[-1].best_params, CFG, wf_result=res)
    assert h.clean
    assert h.start == hold.index[0] and h.end == hold.index[-1]
    assert h.n_bars == len(hold)


def test_warmup_is_trimmed_from_equity():
    df = random_walk()
    dev, hold = split_holdout(df, "2022-12-01")
    h = holdout_test(dev, hold, lambda p: MACrossoverLS(**p),
                     {"fast_period": 10, "slow_period": 50, "stop_pct": 0.08}, CFG)
    assert len(h.equity) == len(hold)
    assert h.equity.index[0] == hold.index[0]


def test_ruined_holdout_reports_minus_100_not_nan():
    n = 1000
    closes = [20000.0]
    for i in range(1, n):
        closes.append(closes[-1] * (0.97 if 700 <= i < 800 else 1.0))
    c = np.array(closes)
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz=UTC)
    df = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c}, index=idx)
    dev, hold = split_holdout(df, str(idx[680].date()))
    h = holdout_test(dev, hold, lambda p: AlwaysLong(), {},
                     WFConfig(warmup_bars=10, leverage=5.0))
    assert h.total_return == -100.0
    assert h.max_dd == -100.0
    for v in (h.cagr, h.sharpe):
        assert not np.isnan(v)
    assert h.liquidations >= 1
    assert "nan" not in h.summary().lower()
