"""
Two-panel backtest chart: equity curve (top) + drawdown underwater plot (bottom).
Entry and exit trades are marked on the equity panel.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from engine.broker import Trade


def plot_results(
    equity: pd.Series,
    trades: list[Trade],
    save_path: str = "backtest_results.png",
    show: bool = True,
) -> None:
    """
    Save (and optionally display) a two-panel backtest chart.

    Panel 1 (top):   equity curve with entry (▲) and exit (▼) markers
    Panel 2 (bottom): drawdown curve (underwater plot)
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # ── Panel 1: Equity curve ──────────────────────────────────────────────
    ax1.plot(equity.index, equity.values, color="steelblue", linewidth=1.5, label="Equity")

    # Mark entries (green up-triangle) and exits (red down-triangle)
    entry_times = [t.entry_time for t in trades if t.entry_time in equity.index or True]
    entry_equities = [float(equity.asof(t.entry_time)) for t in trades]
    if entry_times:
        ax1.scatter(
            entry_times, entry_equities,
            marker="^", color="green", s=60, zorder=5, label="Entry",
        )

    exit_times = [t.exit_time for t in trades if t.exit_time is not None]
    exit_equities = [float(equity.asof(et)) for et in exit_times]
    if exit_times:
        ax1.scatter(
            exit_times, exit_equities,
            marker="v", color="red", s=60, zorder=5, label="Exit",
        )

    ax1.set_ylabel("Equity (USDT)")
    ax1.set_title("Backtest Results")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100.0

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.35)
    ax2.plot(drawdown.index, drawdown.values, color="crimson", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
