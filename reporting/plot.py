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
from datetime import datetime

from engine.broker import Trade
from analysis.levels import find_swing_points, compute_fib_levels, fit_trendline

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
    show: bool = True,
    show_levels: bool = False,
    swing_lookback: int = 8,  # see note below: 8 for rendering, NOT the detector's 5
) -> str:
    """
    Save a three-panel TradingView-style dark chart and return its path.

    Filename is always backtest_YYYY-MM-DD_HHMMSS.png (timestamped, never
    overwrites a previous run). Callers that want a stable filename (e.g. for
    the README) should copy the returned path themselves.

    Panel 1 (top):    BTC 4h price
    Panel 2 (middle): equity curve
    Panel 3 (bottom): drawdown underwater plot

    show_levels (default False — existing behavior unchanged): overlay the
    analysis/levels.py primitives onto the price panel — swing-point markers,
    Fibonacci retracement lines, and support/resistance trendlines. Low-
    confidence trendlines (R^2 < 0.5) are drawn at reduced opacity and dashed,
    not hidden.

    swing_lookback (default 8) — DELIBERATELY different from
    find_swing_points()'s own default of 5. These two numbers serve two
    different purposes and are not an inconsistency:
        * 5  = the detector's default: maximum sensitivity, for a script or UI
               that wants every minor pivot.
        * 8  = the chart-rendering default chosen here: fewer, more SIGNIFICANT
               swings, which produces a legible (non-cramped) Fibonacci grid and
               a clear high- vs low-confidence trendline contrast. At lookback 5
               the "most recent high + most recent low" Fib pair is often two
               adjacent minor pivots, collapsing the grid into a thin band.
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

    # Find the drawdown axis by y-value sign: drawdown is always <= 0, while the
    # price and equity axes have large positive values. Numeric indexing is fragile
    # because candle mode adds scatter twinx axes that extend the axes list unpredictably.
    dd_ax = None
    for ax in axes:
        lines = ax.get_lines()
        if not lines:
            continue
        valid = lines[0].get_ydata()
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0 and valid.max() <= 0.001:
            dd_ax = ax
            break
    if dd_ax is not None:
        dd_ax.fill_between(
            dd_ax.get_lines()[0].get_xdata(),
            dd_ax.get_lines()[0].get_ydata(),
            0,
            color="#ef5350",
            alpha=0.25,
        )

    # ── Optional technical-analysis overlay on the price panel (axes[0]) ──
    if show_levels:
        _overlay_levels(axes[0], data, swing_lookback)

    ts_path = f"backtest_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.png"

    fig.savefig(ts_path, dpi=150, bbox_inches="tight", facecolor="#131722")
    print(f"Chart saved: {ts_path}  [{n_bars:,} bars, {mode_label} mode]")

    if show:
        plt.show()
    plt.close(fig)
    return ts_path


# ---------------------------------------------------------------------------
# Technical-analysis overlay (swing points / Fibonacci / trendlines)
# ---------------------------------------------------------------------------

# Overlay palette
_SWING_HIGH_COLOR = "#f0b90b"   # gold
_SWING_LOW_COLOR = "#b39ddb"    # light purple
_FIB_COLOR = "#d4a017"          # amber
_SUPPORT_COLOR = "#26a69a"      # teal
_RESISTANCE_COLOR = "#ff7f7f"   # coral


def _overlay_levels(ax, data: pd.DataFrame, lookback: int) -> None:
    """
    Draw swing markers, Fibonacci levels, and trendlines onto the price axis.

    All artists are positioned against mplfinance's integer x-axis (bar
    position 0..n-1), which is exactly the `pos` column returned by
    find_swing_points for this same DataFrame — so they align with the candles
    / price line without any timestamp-to-pixel guessing.
    """
    n = len(data)
    swings = find_swing_points(data, lookback=lookback)
    if swings.is_empty:
        return

    # ── Swing-point markers ────────────────────────────────────────────────
    highs, lows = swings.highs, swings.lows
    if not highs.empty:
        ax.scatter(highs["pos"], highs["price"], marker="v", s=26,
                   color=_SWING_HIGH_COLOR, edgecolors="none", zorder=5)
    if not lows.empty:
        ax.scatter(lows["pos"], lows["price"], marker="^", s=26,
                   color=_SWING_LOW_COLOR, edgecolors="none", zorder=5)

    # ── Fibonacci retracement lines ────────────────────────────────────────
    fib = compute_fib_levels(data, swings)
    if fib is not None:
        for ratio, price in fib.levels.items():
            ax.axhline(price, linestyle="--", linewidth=0.8,
                       color=_FIB_COLOR, alpha=0.55, zorder=3)
            ax.text(n - 1, price, f" {ratio:.1%}  {price:,.0f}",
                    color=_FIB_COLOR, fontsize=7, va="center", ha="right",
                    alpha=0.9, zorder=6)

    # ── Trendlines (support = teal, resistance = coral) ────────────────────
    for side, color in (("support", _SUPPORT_COLOR),
                        ("resistance", _RESISTANCE_COLOR)):
        tl = fit_trendline(data, swings, side=side)
        if tl is None:
            continue
        # Low-confidence fits are de-emphasized (faint + dashed), not dropped.
        alpha = 0.30 if tl.low_confidence else 0.95
        lw = 1.0 if tl.low_confidence else 1.7
        ls = (0, (4, 3)) if tl.low_confidence else "-"
        ax.plot([tl.x_start, tl.x_end], [tl.y_start, tl.y_end],
                color=color, alpha=alpha, linewidth=lw, linestyle=ls, zorder=4)
        tag = f"{side} R²={tl.r_squared:.2f}"
        if tl.low_confidence:
            tag += " (low-conf)"
        # Anchor the label at the line's drawn end, clamped inside the frame and
        # right-aligned when it sits at the right edge so the text reads leftward
        # into the chart instead of overflowing off-screen.
        label_x = min(tl.x_end, n - 1)
        ha = "right" if label_x >= n - 1 - 1e-9 else "left"
        ax.text(label_x, tl.y_end, tag + ("  " if ha == "right" else ""),
                color=color, fontsize=7, va="center", ha=ha,
                alpha=max(alpha, 0.7), zorder=6)
