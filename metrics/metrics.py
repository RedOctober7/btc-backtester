"""
Performance metrics for the backtesting engine.

Annualization note: all per-bar statistics use 2190 bars/year for 4h bars
(6 bars/day × 365 days; crypto markets never close). Using 252 (equity trading days)
or 365 without the 6× multiplier would understate Sharpe and vol.

Sharpe note: computed on the FULL per-bar return series including flat bars
(when the strategy holds no position). The "trade-bars-only" version looks
better on paper but is dishonest — flat time carries real opportunity cost.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from engine.broker import Trade

# 6 bars/day × 365 days — crypto runs continuously, 24/7/365
BARS_PER_YEAR = 2190


# ---------------------------------------------------------------------------
# Individual metric functions (all return native Python numbers for easy testing)
# ---------------------------------------------------------------------------

def total_return(equity: pd.Series) -> float:
    """Total percentage return: (final / initial - 1) × 100."""
    return (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0


def cagr(equity: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    Converts total return to an annualized rate using the number of 4h bars elapsed.
    """
    n_years = len(equity) / BARS_PER_YEAR
    return ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0) * 100.0


def annualized_volatility(equity: pd.Series) -> float:
    """Annualized standard deviation of per-bar returns, as a percentage."""
    returns = equity.pct_change().dropna()
    return float(returns.std() * math.sqrt(BARS_PER_YEAR) * 100.0)


def sharpe_ratio(equity: pd.Series) -> float:
    """
    Annualized Sharpe ratio. Risk-free rate = 0 (assumed; documented here).
    Full-series computation: includes all bars, not just bars inside a trade.
    Annualization factor: sqrt(2190) for 4h bars.
    """
    returns = equity.pct_change().dropna()
    std = float(returns.std())
    if std == 0.0:
        return 0.0
    return float((returns.mean() / std) * math.sqrt(BARS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> tuple[float, Optional[pd.Timedelta]]:
    """
    Maximum peak-to-trough drawdown as a negative percentage, and its duration
    (time from the peak to the trough, not to recovery).
    """
    running_max = equity.cummax()
    dd_series = (equity - running_max) / running_max * 100.0
    max_dd = float(dd_series.min())

    # Find duration: from the last peak before the trough to the trough itself
    trough_time = dd_series.idxmin()
    peak_time = running_max.loc[:trough_time].idxmax()
    duration: Optional[pd.Timedelta] = None
    if peak_time != trough_time:
        duration = trough_time - peak_time

    return max_dd, duration


def exposure(in_position: pd.Series) -> float:
    """Percentage of bars where the strategy held a position."""
    return float(in_position.mean() * 100.0)


def trade_stats(trades: list[Trade]) -> dict:
    """Win rate, avg win/loss, profit factor computed from the trade blotter."""
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "profit_factor": float("nan"),
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100.0,
        "avg_win": sum(wins) / len(wins) if wins else float("nan"),
        "avg_loss": sum(losses) / len(losses) if losses else float("nan"),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Aggregate: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(result) -> dict:
    """
    Compute every metric from a BacktestResult.
    Returns a flat dict: metric_name -> value.
    """
    equity = result.equity
    trades = result.trades
    in_position = result.in_position
    stats = trade_stats(trades)
    dd, dd_duration = max_drawdown(equity)

    return {
        "Total Return (%)": total_return(equity),
        "CAGR (%)": cagr(equity),
        "Annualized Volatility (%)": annualized_volatility(equity),
        "Sharpe Ratio": sharpe_ratio(equity),
        "Max Drawdown (%)": dd,
        "Max Drawdown Duration": str(dd_duration) if dd_duration else "N/A",
        "Number of Trades": stats["num_trades"],
        "Win Rate (%)": stats["win_rate"],
        "Avg Win (USDT)": stats["avg_win"],
        "Avg Loss (USDT)": stats["avg_loss"],
        "Profit Factor": stats["profit_factor"],
        "Exposure (%)": exposure(in_position),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_DESCRIPTIONS = {
    "Total Return (%)":         "overall gain/loss over the full period",
    "CAGR (%)":                 "annualized compounded return",
    "Annualized Volatility (%)":"std dev of per-bar returns * sqrt(2190), as %",
    "Sharpe Ratio":             "risk-adjusted return; full-series, annualized, rf=0",
    "Max Drawdown (%)":         "worst peak-to-trough equity decline",
    "Max Drawdown Duration":    "time from peak to trough (not to recovery)",
    "Number of Trades":         "total completed round-trips",
    "Win Rate (%)":             "% of trades with positive PnL",
    "Avg Win (USDT)":           "mean PnL of winning trades",
    "Avg Loss (USDT)":          "mean PnL of losing trades (negative)",
    "Profit Factor":            "gross profit / gross loss; >1 = profitable",
    "Exposure (%)":             "% of bars where strategy held a position",
}


def print_metrics_table(metrics: dict) -> None:
    """Print metrics as a clean aligned table with descriptions."""
    width = 74
    print("\n" + "=" * width)
    print(f"  {'METRIC':<32} {'VALUE':>10}   DESCRIPTION")
    print("-" * width)
    for name, value in metrics.items():
        desc = _DESCRIPTIONS.get(name, "")
        if isinstance(value, float):
            if abs(value) == float("inf") or (value != value):  # inf or NaN
                formatted = str(value)
            else:
                formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        print(f"  {name:<32} {formatted:>10}   {desc}")
    print("=" * width)


def print_trade_blotter(trades: list[Trade]) -> None:
    """Print one row per trade with entry, exit, PnL, and exit reason."""
    if not trades:
        print("  (no trades)")
        return

    header = (
        f"  {'#':>3}  {'Entry Date':<12} {'Exit Date':<12} "
        f"{'Entry $':>10} {'Exit $':>10} "
        f"{'Size BTC':>10} {'Fees $':>8} {'PnL $':>10} "
        f"{'Return %':>9} {'Bars':>5}  Reason"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, t in enumerate(trades, 1):
        entry_date = t.entry_time.strftime("%Y-%m-%d") if t.entry_time else "?"
        exit_date = t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "open"
        bars = str(t.bars_held) if t.bars_held is not None else "?"
        print(
            f"  {i:>3}  {entry_date:<12} {exit_date:<12} "
            f"{t.entry_price:>10.2f} {(t.exit_price or 0):>10.2f} "
            f"{t.size:>10.4f} {t.fees_paid:>8.2f} {t.pnl:>10.2f} "
            f"{t.return_pct:>9.2f} {bars:>5}  {t.exit_reason}"
        )
