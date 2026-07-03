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
    """Score a fold's backtest during IS optimization; higher = better.
    Default is annualized Sharpe. For crypto, Calmar (CAGR / |maxDD|) is arguably
    the better target since it punishes the deep drawdowns Sharpe waves through —
    swap it in via WFConfig.score if you want that."""
    return M.sharpe_ratio(r.equity)


@dataclass
class WFConfig:
    # window geometry, in 4h bars
    is_bars: int = 1080          # in-sample window  (~6 months at 6 bars/day)
    oos_bars: int = 180          # out-of-sample window (~1 month)
    warmup_bars: int = 200       # >= longest indicator lookback (slow_period, etc.).
                                 # Prepended to each OOS slice so SMAs can warm up;
                                 # leave too low and long-lookback folds read as flat.
    anchored: bool = False       # False = rolling (slides); True = anchored (IS grows).
                                 # Rolling is the right default for BTC — it lets the
                                 # optimizer forget dead regimes.
    # broker settings — passed straight through to your run_backtest
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001
    slippage_bps: float = 0.0
    position_fraction: float = 1.0
    leverage: float = 1.0             # 1.0 = spot-equivalent. Start every new
                                       # strategy's walk-forward at 1.0 first —
                                       # find out if it has edge BEFORE adding
                                       # leverage risk on top of an unproven signal.
    funding_rate_8h: float = 0.0      # set a real resting rate (~0.0001) whenever
                                       # leverage > 1, or OOS results overstate holds.
    maintenance_margin_rate: float = 0.005
    liquidation_penalty_bps: float = 0.0
    # IS optimization target
    score: Callable[[BacktestResult], float] = field(default=_default_score)
    score_name: str = "Sharpe"


# ══════════════════════════════════════════════════════════════════════════
# Small return helpers (fractions, so the WFE math stays honest about units)
# ══════════════════════════════════════════════════════════════════════════

def _return_frac(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def _safe_sharpe(equity: pd.Series) -> float:
    """Sharpe that returns 0.0 instead of NaN on degenerate curves (all-zero
    after a wipeout, too short, or zero-variance). A dead account has no
    risk-adjusted return; 0.0 with the wipeout reported elsewhere is honest,
    NaN in a metrics table is just noise."""
    eq = equity.dropna()
    if len(eq) < 2 or float(eq.iloc[0]) <= 0.0:
        return 0.0
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    std = float(rets.std()) if len(rets) else 0.0
    if not np.isfinite(std) or std == 0.0:
        return 0.0
    return float((rets.mean() / std) * np.sqrt(BARS_PER_YEAR))


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
    oos_start: int         # also = is_end (exclusive)
    oos_end: int           # exclusive
    best_params: dict
    is_score: float        # optimization score on IS (clean — this picked the params)
    is_return: float       # fraction, over IS window
    oos_return: float      # fraction, over trimmed OOS window (honest)
    oos_sharpe: float      # Sharpe on trimmed OOS equity
    oos_trades: int
    oos_liquidations: int  # count of OOS trades that ended in forced liquidation
    warmup_wiped: bool     # True if the account was liquidated to zero DURING the
                           # warmup bars, before the OOS window began. The fold is
                           # then UNMEASURABLE — an artifact of trading through
                           # warmup, not a real OOS outcome — and is excluded from
                           # stitching. Any nonzero count is a red flag that this
                           # leverage level dies on this data's volatility.
    oos_equity: pd.Series  # trimmed to the true OOS region


def _make_folds(n: int, cfg: WFConfig) -> Iterator[tuple[int, int, int, int]]:
    """Yield (index, is_start, oos_start, oos_end). OOS windows are contiguous and
    non-overlapping so their equity curves stitch into one clean series."""
    oos_start, i = cfg.is_bars, 0
    while oos_start + cfg.oos_bars <= n:
        oos_end = oos_start + cfg.oos_bars
        is_start = 0 if cfg.anchored else oos_start - cfg.is_bars
        yield i, is_start, oos_start, oos_end
        i += 1
        oos_start = oos_end


def _combos(grid: dict[str, list]) -> list[dict]:
    if not grid:
        raise ValueError("param_grid is empty — give at least one param to sweep.")
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _run(strategy: Strategy, data: pd.DataFrame, cfg: WFConfig) -> BacktestResult:
    return run_backtest(
        strategy, data,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_bps=cfg.slippage_bps,
        position_fraction=cfg.position_fraction,
        leverage=cfg.leverage,
        funding_rate_8h=cfg.funding_rate_8h,
        maintenance_margin_rate=cfg.maintenance_margin_rate,
        liquidation_penalty_bps=cfg.liquidation_penalty_bps,
    )


def _optimize(
    data: pd.DataFrame, is_start: int, oos_start: int,
    factory: Callable[[dict], Strategy], grid: dict, cfg: WFConfig,
) -> tuple[float, dict, BacktestResult]:
    df_is = data.iloc[is_start:oos_start]
    best = None  # (score, params, result)
    for params in _combos(grid):
        res = _run(factory(params), df_is, cfg)
        s = cfg.score(res)
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = -np.inf   # e.g. a combo that never trades — never let it win
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
    """
    data              : OHLCV DataFrame, UTC-indexed ascending (same shape your
                        engine already eats).
    strategy_factory  : params dict -> Strategy instance,
                        e.g. lambda p: MACrossover(**p)
    param_grid        : {param_name: [values]}; the cartesian product is swept on
                        every IS window. Keep it lean — combos x folds full runs.
    """
    n = len(data)
    need = cfg.is_bars + cfg.oos_bars
    if n < need:
        raise ValueError(f"Need >= {need} bars (IS+OOS), got {n}.")

    folds: list[Fold] = []
    for i, is_start, oos_start, oos_end in _make_folds(n, cfg):
        is_score, best_params, is_res = _optimize(
            data, is_start, oos_start, strategy_factory, param_grid, cfg
        )

        # OOS run gets warmup history prepended, then we trim back to the true
        # OOS region. Without warmup, any lookback > oos_bars reads as flat.
        run_start = max(0, oos_start - cfg.warmup_bars)
        oos_res = _run(strategy_factory(best_params), data.iloc[run_start:oos_end], cfg)

        # Trim positionally, not by timestamp: the engine marks equity once per
        # bar, so the OOS region is exactly the tail after the warmup offset.
        # (Label matching would break silently if your index is tz-naive while
        # the broker force-localizes equity to UTC.)
        oos_offset = oos_start - run_start
        oos_equity = oos_res.equity.iloc[oos_offset:]

        # trades realized (exited) inside the OOS window, not the warmup tail
        oos_window_trades = [
            t for t in oos_res.trades
            if t.exit_bar_idx is not None and t.exit_bar_idx >= oos_offset
        ]
        oos_trades = len(oos_window_trades)
        oos_liquidations = sum(1 for t in oos_window_trades if t.exit_reason == "liquidation")
        warmup_wiped = len(oos_equity) > 0 and float(oos_equity.iloc[0]) <= 0.0

        folds.append(Fold(
            index=i, is_start=is_start, oos_start=oos_start, oos_end=oos_end,
            best_params=best_params,
            is_score=float(is_score),
            is_return=_return_frac(is_res.equity),
            oos_return=_return_frac(oos_equity),
            oos_sharpe=_safe_sharpe(oos_equity),
            oos_trades=oos_trades,
            oos_liquidations=oos_liquidations,
            warmup_wiped=warmup_wiped,
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
        """Compound the OOS segments into one continuous curve (each fold picks up
        where the last left off).

        RUIN SEMANTICS (matters at leverage): if the compounded capital reaches
        ~zero, the account is dead and STAYS dead — the rest of the curve
        flatlines at zero. Without this, a fold whose equity starts at 0 (e.g.
        liquidated during its warmup prepend) divides by zero and poisons the
        entire stitched curve and every downstream metric with NaN. A real
        trader who blew up does not get a fresh bankroll next fold; neither
        does this curve.
        """
        EPS = 1e-9
        pieces, capital = [], float(self.cfg.initial_capital)
        self.ruined = False           # exposed so summary() can shout about it
        for f in self.folds:
            eq = f.oos_equity.dropna()
            if eq.empty:
                continue
            if self.ruined or capital <= EPS or eq.iloc[0] <= EPS:
                # Account is dead (or this fold began dead): flatline at zero
                # for this fold's timestamps instead of dividing by ~0.
                self.ruined = True
                pieces.append(pd.Series(0.0, index=eq.index, name="equity"))
                capital = 0.0
                continue
            normed = eq / eq.iloc[0] * capital
            pieces.append(normed)
            capital = float(normed.iloc[-1])
            if capital <= EPS:
                self.ruined = True
        return pd.concat(pieces) if pieces else pd.Series(dtype=float, name="equity")

    def _compute_oos_metrics(self) -> None:
        eq = self.oos_equity
        if len(eq) < 2:
            self.total_return = self.cagr = self.sharpe = self.max_dd = float("nan")
            return
        if getattr(self, "ruined", False):
            # Total loss: report it plainly instead of letting a zero tail
            # produce division-by-zero or misleading annualized math. Sharpe is
            # measured on the surviving prefix (the curve before ruin); if the
            # account died before producing a measurable prefix, 0.0 — the
            # RUINED banner in summary() carries the real story, a NaN here
            # would just break tables.
            self.total_return = -100.0
            self.cagr = -100.0
            self.sharpe = _safe_sharpe(eq[eq > 0])
            self.max_dd = -100.0
            return
        self.total_return = M.total_return(eq)     # percent
        self.cagr = M.cagr(eq)                      # percent
        self.sharpe = M.sharpe_ratio(eq)            # ratio
        self.max_dd, _ = M.max_drawdown(eq)         # percent

    @property
    def wfe(self) -> float:
        """Aggregate walk-forward efficiency: annualized OOS return vs mean
        annualized IS return. Aggregate, NOT a mean of per-fold ratios — that
        explodes on any fold whose IS return sits near zero. NaN if IS wasn't
        profitable on average (nothing meaningful to divide by)."""
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
        """Per param, fraction of fold-to-fold transitions where the winning value
        changed. High churn = optimizer chasing noise = overfit smell."""
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
                "oos_liq": f.oos_liquidations,
                "wu_wiped": f.warmup_wiped,
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
             f"folds={len(self.folds)}  target={c.score_name}  leverage={c.leverage}x",
             "",
             "Stitched OOS (the honest number):",
             f"  return  {self.total_return:+.1f}%    CAGR  {self.cagr:+.1f}%",
             f"  sharpe  {self.sharpe:.2f}       maxDD  {self.max_dd:.1f}%",
             ""]
        total_liq = sum(f.oos_liquidations for f in self.folds)
        wiped = sum(1 for f in self.folds if f.warmup_wiped)
        if getattr(self, "ruined", False):
            L.append("*** ACCOUNT RUINED: stitched capital hit zero mid-run — total loss. ***")
            if wiped:
                L.append(f"    ({wiped}/{len(self.folds)} folds died during their warmup bars —")
                L.append(f"    this leverage level does not survive this data's volatility.)")
        if c.leverage > 1.0:
            flag = "  <-- leverage is amplifying losses, not revealing edge" if total_liq > 0 else ""
            L.append(f"Liquidations in OOS: {total_liq}{flag}")
        L.append(f"Walk-Forward Efficiency: {self.wfe:.0%}  ->  {self._verdict(self.wfe)}")
        churn = self.param_churn()
        if churn:
            worst = max(churn.values())
            tag = " (stable)" if worst <= 0.3 else " (unstable — flag)" if worst > 0.6 else ""
            L.append("Param churn: " + ", ".join(f"{k} {v:.0%}" for k, v in churn.items()) + tag)
        return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
# Usage / smoke test.  `python walk_forward.py`
# Swap the synthetic block for your real OHLCV loader and you're done.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from strategies.ma_crossover_ls import MACrossoverLS

    # ---- REAL USAGE looks like this -------------------------------------
    #   data = load_your_4h_ohlcv()          # UTC-indexed, ascending
    #   factory = lambda p: MACrossoverLS(**p)
    #   grid = {"fast_period": [20, 50, 80], "slow_period": [150, 200, 250],
    #           "stop_pct": [0.06, 0.08, 0.12]}
    #
    #   Step 1 — ALWAYS run leverage=1 first. This answers "does long/short
    #   direction add edge over long-only" before any leverage risk is added:
    #   res = walk_forward(data, factory, grid,
    #                      WFConfig(is_bars=1080, oos_bars=180, warmup_bars=250,
    #                               leverage=1.0))
    #
    #   Step 2 — only if step 1 shows real WFE, test leverage with a realistic
    #   funding rate (leverage without funding overstates every result):
    #   res_lev = walk_forward(data, factory, grid,
    #                          WFConfig(is_bars=1080, oos_bars=180, warmup_bars=250,
    #                                   leverage=3.0, funding_rate_8h=0.0001,
    #                                   liquidation_penalty_bps=25))
    #   print(res.summary()); print(res.to_dataframe().to_string(index=False))
    # ---------------------------------------------------------------------

    from data.loader import load_candles
    data = load_candles("BTCUSDT", "4h", "2022-01-01", "2026-07-02")

    factory = lambda p: MACrossoverLS(**p)
    grid = {"fast_period": [20, 50], "slow_period": [150, 200], "stop_pct": [0.08]}

    print("=" * 70)
    print("LEVERAGE = 1x  (isolate whether long/short direction has edge)")
    print("=" * 70)
    res1 = walk_forward(
        data, factory, grid,
        WFConfig(is_bars=1080, oos_bars=180, warmup_bars=200, leverage=1.0),
    )
    print(res1.summary())
    print()
    print(res1.to_dataframe().to_string(index=False))

    print()
    print("=" * 70)
    print("LEVERAGE = 3x, funding 0.0001/8h  (only meaningful if 1x showed edge)")
    print("=" * 70)
    res3 = walk_forward(
        data, factory, grid,
        WFConfig(is_bars=1080, oos_bars=180, warmup_bars=200,
                 leverage=3.0, funding_rate_8h=0.0001, liquidation_penalty_bps=25),
    )
    print(res3.summary())
    print()
    print(res3.to_dataframe().to_string(index=False))
