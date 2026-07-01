"""
walk_forward.py — Walk-forward analysis for btc-backtester.

Drop this at the project root, next to engine/ and strategies/.

Optimize strategy params on an in-sample (IS) window, test the winning params on
the very next out-of-sample (OOS) window the optimizer never touched, roll
forward, and stitch the OOS segments into one continuous equity curve. The
stitched OOS curve is your honest forward estimate. Walk-Forward Efficiency (WFE)
tells you whether the IS edge survives contact with unseen data or was just
curve-fit to noise — the thing a single glossy backtest can't show you.

Numbers come from your own engine.metrics (BARS_PER_YEAR=2190, rf=0 Sharpe), so
they line up with the rest of the system exactly.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from engine.engine import run_backtest, BacktestResult
from metrics import metrics as M
from strategies.base import Strategy

BARS_PER_YEAR = M.BARS_PER_YEAR  # 2190 for 4h crypto bars


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

def _default_score(r: BacktestResult) -> float:
    return M.sharpe_ratio(r.equity)


@dataclass
class WFConfig:
    is_bars: int = 1080
    oos_bars: int = 180
    warmup_bars: int = 200
    anchored: bool = False
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001
    slippage_bps: float = 0.0
    position_fraction: float = 1.0
    score: Callable[[BacktestResult], float] = field(default=_default_score)
    score_name: str = "Sharpe"


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _return_frac(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def _annualize(ret_frac: float, n_bars: int) -> float:
    if n_bars <= 0:
        return 0.0
    return (1.0 + ret_frac) ** (BARS_PER_YEAR / n_bars) - 1.0


# ══════════════════════════════════════════════════════════════════════════
# Folds
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Fold:
    index: int
    is_start: int
    oos_start: int
    oos_end: int
    best_params: dict
    is_score: float
    is_return: float
    oos_return: float
    oos_sharpe: float
    oos_trades: int
    oos_equity: pd.Series


def _make_folds(n: int, cfg: WFConfig) -> Iterator[tuple[int, int, int, int]]:
    oos_start, i = cfg.is_bars, 0
    while oos_start + cfg.oos_bars <= n:
        oos_end = oos_start + cfg.oos_bars
        is_start = 0 if cfg.anchored else oos_start - cfg.is_bars
        yield i, is_start, oos_start, oos_end
        i += 1
        oos_start = oos_end


def _combos(grid: dict[str, list]) -> list[dict]:
    if not grid:
        raise ValueError("param_grid is empty.")
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _run(strategy: Strategy, data: pd.DataFrame, cfg: WFConfig) -> BacktestResult:
    return run_backtest(
        strategy, data,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_bps=cfg.slippage_bps,
        position_fraction=cfg.position_fraction,
    )


def _optimize(
    data: pd.DataFrame, is_start: int, oos_start: int,
    factory: Callable[[dict], Strategy], grid: dict, cfg: WFConfig,
) -> tuple[float, dict, BacktestResult]:
    df_is = data.iloc[is_start:oos_start]
    best = None
    for params in _combos(grid):
        res = _run(factory(params), df_is, cfg)
        s = cfg.score(res)
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = -np.inf
        if best is None or s > best[0]:
            best = (s, params, res)
    return best


# ══════════════════════════════════════════════════════════════════════════
# Main driver
# ══════════════════════════════════════════════════════════════════════════

def walk_forward(
    data: pd.DataFrame,
    strategy_factory: Callable[[dict], Strategy],
    param_grid: dict[str, list],
    cfg: WFConfig = WFConfig(),
) -> "WFResult":
    n = len(data)
    need = cfg.is_bars + cfg.oos_bars
    if n < need:
        raise ValueError(f"Need >= {need} bars (IS+OOS), got {n}.")

    folds: list[Fold] = []
    for i, is_start, oos_start, oos_end in _make_folds(n, cfg):
        is_score, best_params, is_res = _optimize(
            data, is_start, oos_start, strategy_factory, param_grid, cfg
        )

        run_start = max(0, oos_start - cfg.warmup_bars)
        oos_res = _run(strategy_factory(best_params), data.iloc[run_start:oos_end], cfg)

        oos_offset = oos_start - run_start
        oos_equity = oos_res.equity.iloc[oos_offset:]

        oos_trades = sum(
            1 for t in oos_res.trades
            if t.exit_bar_idx is not None and t.exit_bar_idx >= oos_offset
        )

        folds.append(Fold(
            index=i, is_start=is_start, oos_start=oos_start, oos_end=oos_end,
            best_params=best_params,
            is_score=float(is_score),
            is_return=_return_frac(is_res.equity),
            oos_return=_return_frac(oos_equity),
            oos_sharpe=M.sharpe_ratio(oos_equity) if len(oos_equity) > 1 else 0.0,
            oos_trades=oos_trades,
            oos_equity=oos_equity,
        ))

    if not folds:
        raise ValueError("No complete folds — shrink is_bars/oos_bars.")
    return WFResult(folds, cfg)


# ══════════════════════════════════════════════════════════════════════════
# Result
# ══════════════════════════════════════════════════════════════════════════

class WFResult:
    def __init__(self, folds: list[Fold], cfg: WFConfig):
        self.folds = folds
        self.cfg = cfg
        self.oos_equity = self._stitch()
        self._compute_oos_metrics()

    def _stitch(self) -> pd.Series:
        pieces, capital = [], float(self.cfg.initial_capital)
        for f in self.folds:
            eq = f.oos_equity.dropna()
            if eq.empty:
                continue
            pieces.append(eq / eq.iloc[0] * capital)
            capital = pieces[-1].iloc[-1]
        return pd.concat(pieces) if pieces else pd.Series(dtype=float, name="equity")

    def _compute_oos_metrics(self) -> None:
        eq = self.oos_equity
        if len(eq) < 2:
            self.total_return = self.cagr = self.sharpe = self.max_dd = float("nan")
            return
        self.total_return = M.total_return(eq)
        self.cagr = M.cagr(eq)
        self.sharpe = M.sharpe_ratio(eq)
        self.max_dd, _ = M.max_drawdown(eq)

    @property
    def wfe(self) -> float:
        is_ann = [
            _annualize(f.is_return, f.oos_start - f.is_start) for f in self.folds
        ]
        if not is_ann:
            return float("nan")
        is_base = float(np.mean(is_ann))
        if is_base <= 0:
            return float("nan")
        return (self.cagr / 100.0) / is_base

    def param_churn(self) -> dict[str, float]:
        keys = list(self.folds[0].best_params) if self.folds else []
        if len(self.folds) < 2:
            return {k: 0.0 for k in keys}
        out = {}
        for k in keys:
            changes = sum(
                self.folds[i].best_params.get(k) != self.folds[i - 1].best_params.get(k)
                for i in range(1, len(self.folds))
            )
            out[k] = changes / (len(self.folds) - 1)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            row = {
                "fold": f.index,
                f"is_{self.cfg.score_name}": round(f.is_score, 3),
                "is_ret%": round(f.is_return * 100, 2),
                "oos_ret%": round(f.oos_return * 100, 2),
                "oos_sharpe": round(f.oos_sharpe, 2),
                "oos_trades": f.oos_trades,
            }
            row.update(f.best_params)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _verdict(wfe: float) -> str:
        if np.isnan(wfe):
            return "undefined (IS wasn't profitable on average — nothing to hold up)"
        if wfe >= 0.6:
            return "robust"
        if wfe >= 0.4:
            return "marginal — tighten the grid or widen OOS before trusting it"
        return "OVERFIT — OOS gives back the IS edge, don't trade this"

    def summary(self) -> str:
        c = self.cfg
        L = ["Walk-Forward Analysis",
             f"  {'anchored' if c.anchored else 'rolling'}  "
             f"IS={c.is_bars} OOS={c.oos_bars} warmup={c.warmup_bars} bars  "
             f"folds={len(self.folds)}  target={c.score_name}",
             "",
             "Stitched OOS (the honest number):",
             f"  return  {self.total_return:+.1f}%    CAGR  {self.cagr:+.1f}%",
             f"  sharpe  {self.sharpe:.2f}       maxDD  {self.max_dd:.1f}%",
             ""]
        L.append(f"Walk-Forward Efficiency: {self.wfe:.0%}  ->  {self._verdict(self.wfe)}")
        churn = self.param_churn()
        if churn:
            worst = max(churn.values())
            tag = " (stable)" if worst <= 0.3 else " (unstable — flag)" if worst > 0.6 else ""
            L.append("Param churn: " + ", ".join(f"{k} {v:.0%}" for k, v in churn.items()) + tag)
        return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
# __main__
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from strategies.ma_crossover import MACrossover

    # ---- REAL LOADER (reuses data/loader.py exactly as run.py does) ----
    from data.loader import load_candles
    data = load_candles("BTCUSDT", "4h", "2022-01-01", "2026-06-25")
    # ---- END LOADER ----

    factory = lambda p: MACrossover(**p)
    grid = {"fast_period": [20, 30], "slow_period": [200], "stop_pct": [0.08]}

    # Split before walk_forward so folds never touch holdout
    HOLDOUT_START = "2026-03-01"
    holdout_mask = data.index >= pd.Timestamp(HOLDOUT_START, tz="UTC")
    dev_data     = data[~holdout_mask]
    holdout_data = data[holdout_mask]

    res = walk_forward(
        dev_data, factory, grid,
        WFConfig(is_bars=1080, oos_bars=180, warmup_bars=250),
    )
    print(res.summary())
    print()
    print(res.to_dataframe().to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════
    # TRUE HOLDOUT TEST — one shot, no tuning, never seen by any fold above
    # ══════════════════════════════════════════════════════════════════════
    FIXED_PARAMS = {"fast_period": 20, "slow_period": 200, "stop_pct": 0.08}
    WARMUP = 250

    # Confirm walk_forward() never touched holdout — last fold's oos_end is an
    # integer index into `data`; translate to timestamp for the check.
    last_fold_end_ts = data.index[res.folds[-1].oos_end - 1]
    holdout_first_ts = holdout_data.index[0]

    print()
    print("=" * 68)
    print("TRUE HOLDOUT TEST")
    print(f"  dev_data   : {dev_data.index[0].date()} -> {dev_data.index[-1].date()}  ({len(dev_data)} bars)")
    print(f"  holdout    : {holdout_first_ts.date()} -> {holdout_data.index[-1].date()}  ({len(holdout_data)} bars)")
    print(f"  last fold ended at bar index {res.folds[-1].oos_end - 1} ({last_fold_end_ts.date()})")
    print(f"  holdout starts at {holdout_first_ts.date()} — {'CLEAN (no overlap)' if last_fold_end_ts < holdout_first_ts else 'OVERLAP — not clean'}")
    print(f"  fixed params: {FIXED_PARAMS}")
    print("=" * 68)

    # Prepend warmup bars from dev_data so indicators are warm at holdout[0]
    warmup_slice = dev_data.iloc[-WARMUP:]
    run_data     = pd.concat([warmup_slice, holdout_data])

    cfg_holdout = WFConfig(
        initial_capital=10_000.0, fee_rate=0.001,
        slippage_bps=0.0, position_fraction=1.0,
    )
    holdout_res = _run(factory(FIXED_PARAMS), run_data, cfg_holdout)

    # Strip the warmup portion from equity and trades before reporting
    holdout_equity = holdout_res.equity.iloc[WARMUP:]
    holdout_trades = sum(
        1 for t in holdout_res.trades
        if t.exit_bar_idx is not None and t.exit_bar_idx >= WARMUP
    )

    h_return = M.total_return(holdout_equity)
    h_cagr   = M.cagr(holdout_equity)
    h_sharpe = M.sharpe_ratio(holdout_equity)
    h_dd, _  = M.max_drawdown(holdout_equity)

    print(f"  return     : {h_return:+.2f}%")
    print(f"  CAGR       : {h_cagr:+.2f}%")
    print(f"  Sharpe     : {h_sharpe:.2f}")
    print(f"  max DD     : {h_dd:.2f}%")
    print(f"  trades     : {holdout_trades}")
    print("=" * 68)
