"""
TradingView-style dark three-panel backtest chart powered by mplfinance.

Panel layout (top to bottom):
    1. BTC 4h price  — candlesticks (<=540 bars) or price line (>540 bars)
    2. Equity curve
    3. Drawdown (underwater fill)

Candlestick vs. line rule:
    CANDLE_MODE_MAX_BARS = 540  (90 days x 6 bars/day)
    <= 540 bars: candlesticks + entry/exit triangle markers at actual fill price
    >  540 bars: price line only, no markers
    At multi-year zoom, 17-31 tiny triangles overlap and add noise, not clarity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from engine.broker import Trade

CANDLE_MODE_MAX_BARS = 540  # 90 days x 6 bars/day

# ---------------------------------------------------------------------------
# TradingView dark theme styles
# ---------------------------------------------------------------------------

_BASE_RC = {
    "axes.labelcolor": "#d1d4dc",
    "axes.titlecolor": "#d1d4dc",
    "xtick.color": "#d1d4dc",
    "ytick.color": "#d1d4dc",
    "text.color": "#d1d4dc",
}

# Candlestick mode: green/red candles
_CANDLE_STYLE = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick={"up": "#26a69a", "down": "#ef5350"},
        volume="inherit",
    ),
    facecolor="#131722",
    edgecolor="#2a2e39",
    figcolor="#131722",
    gridcolor="#2a2e39",
    gridstyle="-",
    y_on_right=False,
    rc=_BASE_RC,
)

# Line mode: close price drawn as a light gray/white line
# mplfinance type='line' uses marketcolors.up for the line color
_LINE_STYLE = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(
        up="#d1d4dc",
        down="#d1d4dc",
        edge="inherit",
        wick="inherit",
        volume="inherit",
    ),
    facecolor="#131722",
    edgecolor="#2a2e39",
    figcolor="#131722",
    gridcolor="#2a2e39",
    gridstyle="-",
    y_on_right=False,
    rc=_BASE_RC,
)


def plot_results(
    equity: pd.Series,
    data: pd.DataFrame,
    trades: list[Trade],
    save_path: str = "backtest_results.png",
    show: bool = True,
) -> None:
    """
    Save (and optionally display) a three-panel TradingView-style dark chart.

    Panel 1 (top):    BTC 4h price
    Panel 2 (middle): equity curve
    Panel 3 (bottom): drawdown underwater plot
    """
    n_bars = len(data)
    candle_mode = n_bars <= CANDLE_MODE_MAX_BARS
    style = _CANDLE_STYLE if candle_mode else _LINE_STYLE
    mode_label = "candle" if candle_mode else "line"

    # ── Drawdown (aligned to data.index, same as equity) ──────────────────
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100.0

    # ── addplot: equity curve (panel 1) ───────────────────────────────────
    ap_equity = mpf.make_addplot(
        equity,
        panel=1,
        color="#7eb8f7",
        width=1.5,
        ylabel="Equity (USDT)",
    )

    # ── addplot: drawdown line (panel 2) — fill added manually post-plot ──
    ap_drawdown = mpf.make_addplot(
        drawdown,
        panel=2,
        color="#ef5350",
        width=0.8,
        ylabel="Drawdown (%)",
    )

    addplots = [ap_equity, ap_drawdown]

    # ── Entry/exit markers — candlestick mode only ─────────────────────────
    if candle_mode and trades:
        entry_prices = pd.Series(np.nan, index=data.index, dtype=float)
        exit_prices = pd.Series(np.nan, index=data.index, dtype=float)

        for t in trades:
            # Markers at actual fill price, not signal price — honest representation
            idx = data.index.get_indexer([t.entry_time], method="nearest")[0]
            if idx >= 0:
                entry_prices.iloc[idx] = t.entry_price
            if t.exit_time is not None:
                idx = data.index.get_indexer([t.exit_time], method="nearest")[0]
                if idx >= 0:
                    exit_prices.iloc[idx] = t.exit_price

        if entry_prices.notna().any():
            addplots.append(mpf.make_addplot(
                entry_prices, panel=0, type="scatter",
                markersize=80, marker="^", color="#26a69a",
            ))
        if exit_prices.notna().any():
            addplots.append(mpf.make_addplot(
                exit_prices, panel=0, type="scatter",
                markersize=80, marker="v", color="#ef5350",
            ))

    # ── Render ────────────────────────────────────────────────────────────
    date_range = f"{data.index[0].date()} to {data.index[-1].date()}"
    title = (
        f"\nBTC 4h | SMA(50,200) + 8% stop | {date_range} | "
        f"{n_bars:,} bars [{mode_label} mode]"
    )

    fig, axes = mpf.plot(
        data,
        type="candle" if candle_mode else "line",
        style=style,
        addplot=addplots,
        panel_ratios=(3, 2, 1),
        figsize=(16, 10),
        returnfig=True,
        title=title,
        datetime_format="%Y-%m",
        xrotation=30,
        tight_layout=True,
    )

    # Manual drawdown fill using x-coords from the line mplfinance already drew,
    # which guarantees alignment with the internal axis regardless of how mplfinance
    # maps timestamps to float positions internally.
    dd_ax = axes[2]
    dd_lines = dd_ax.get_lines()
    if dd_lines:
        dd_ax.fill_between(
            dd_lines[0].get_xdata(),
            dd_lines[0].get_ydata(),
            0,
            color="#ef5350",
            alpha=0.25,
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#131722")
    print(f"Chart saved to: {save_path}  [{n_bars:,} bars, {mode_label} mode]")

    if show:
        plt.show()
    plt.close(fig)
