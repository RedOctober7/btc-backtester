"""
analysis/levels.py — standalone technical-analysis primitives.

Three independent functions, each taking an OHLCV DataFrame and returning plain
structured data (frozen dataclasses) — no chart objects, no matplotlib imports:

    find_swing_points(df, lookback)        -> SwingPointResult
    compute_fib_levels(df, swing_points)   -> FibLevels | None
    fit_trendline(df, swing_points, side)  -> Trendline | None

============================================================================
THESE ARE ALGORITHMIC APPROXIMATIONS, NOT GROUND TRUTH
============================================================================
Each function implements ONE standard, defensible method. None of them is "the"
objectively correct way to mark structure on a chart. Two competent human
chartists routinely disagree on where a swing is, which swing pair anchors a
Fibonacci grid, and how to draw a trendline. Treat the output as a reproducible,
documented approximation — useful for visualization and as a starting point —
not as an authoritative read of the market.

============================================================================
LOOKAHEAD BOUNDARY — READ BEFORE REUSING THIS IN TRADING LOGIC
============================================================================
Swing-point detection is CENTERED: a bar is confirmed as a swing only after
`lookback` MORE bars print on its right-hand side. That means a swing at bar i
is not knowable until bar i+lookback. Using these functions inside a strategy's
on_bar() at bar i would leak information from bars i+1..i+lookback into the
decision at bar i — a direct violation of the engine's lookahead firewall.

This module is for VISUALIZATION of historical data only. Do NOT call it from
on_bar() without first re-deriving a causal (right-edge-only) variant that
confirms swings using only past bars. That adaptation is deliberately out of
scope here.

============================================================================
DEPENDENCIES
============================================================================
pandas, numpy only. No imports from engine/, strategies/, or broker — this
package must stay consumable by both run.py and a future UI in the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Standard Fibonacci retracement ratios (the five interior levels).
FIB_RATIOS: tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786)

# Trendline fit-quality cutoff. A linear fit with R^2 below this is still
# returned, but tagged low_confidence=True so the renderer can de-emphasize it.
# 0.5 means "the line explains at least half the variance in the swing prices";
# below that the points are scattered enough that calling it a trend is dubious.
LOW_CONFIDENCE_R2: float = 0.5

# How far past the last contributing swing point a trendline is projected,
# as a fraction of the swing-point span (first..last contributing point).
# 0.20 keeps the projection proportional to the data that produced it: a line
# built across 30 bars of swings extends ~6 bars forward, not indefinitely.
TRENDLINE_PROJECTION_FRAC: float = 0.20

# A swing high/low pair whose range is below this fraction of the price level is
# treated as degenerate (no meaningful Fibonacci grid can be drawn through it).
_DEGENERATE_RANGE_FRAC: float = 1e-6


# ---------------------------------------------------------------------------
# Shared metadata — every result carries this for traceability/reproducibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisMetadata:
    """
    Provenance for one analysis result. Lets a caller distinguish results
    computed with different parameters or over different ranges, so old vs new
    output is never silently conflated.
    """
    method: str                       # algorithm name, e.g. "fractal_swing"
    params: dict                      # exact parameters used
    start: Optional[pd.Timestamp]     # first timestamp of the analyzed range
    end: Optional[pd.Timestamp]       # last timestamp of the analyzed range
    n_bars: int                       # number of bars analyzed


# ---------------------------------------------------------------------------
# 1. Swing points (fractal method)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwingPointResult:
    """
    Detected swing highs and lows.

    points: DataFrame indexed by timestamp with columns
        - price : float  (the swing's high price for 'high', low price for 'low')
        - type  : str    ('high' or 'low')
        - pos   : int    (integer bar position within the analyzed DataFrame;
                          provided for plotting against a positional x-axis and
                          for audit — it is meaningful only relative to the
                          exact df described by metadata)
    """
    points: pd.DataFrame
    metadata: AnalysisMetadata

    @property
    def is_empty(self) -> bool:
        return self.points.empty

    @property
    def highs(self) -> pd.DataFrame:
        """Swing highs only, in chronological order."""
        return self.points[self.points["type"] == "high"]

    @property
    def lows(self) -> pd.DataFrame:
        """Swing lows only, in chronological order."""
        return self.points[self.points["type"] == "low"]


def _empty_points() -> pd.DataFrame:
    return pd.DataFrame(
        {"price": pd.Series(dtype=float),
         "type": pd.Series(dtype=object),
         "pos": pd.Series(dtype=int)}
    )


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> SwingPointResult:
    """
    Detect swing highs and lows using the symmetric fractal method.

    Method (ONE standard, defensible choice — not the only valid one):
        A bar is a swing HIGH if its `high` is STRICTLY greater than the `high`
        of every other bar within `lookback` bars on each side (a symmetric
        2*lookback+1 window centered on the bar). A swing LOW is the inverse on
        `low`. The strict comparison means flat plateaus do not register as
        swings, and each detected swing is the unique extreme of its window.

    Failsafes:
        * If len(df) < lookback*2 + 1 there is not enough data to confirm even
          one centered swing, so an EMPTY result is returned (never an error).
        * The first and last `lookback` bars can never be confirmed (they lack
          bars on one side); the center index range below excludes them
          naturally rather than emitting false edge-of-data swings.

    LOOKAHEAD: centered detection — a swing at bar i is only knowable at bar
    i+lookback. See the module docstring. Visualization-only.

    Default lookback=5 is the DETECTION default: maximum sensitivity, surfacing
    every minor pivot for a script or UI that wants them. Note this is
    intentionally different from the CHART-RENDERING default (8) used by
    reporting.plot.plot_results(show_levels=True): a higher lookback yields
    fewer, more significant swings that render a legible Fibonacci grid and a
    clear high/low-confidence trendline contrast. Two defaults, two purposes —
    not an inconsistency.

    Returns a SwingPointResult; result.is_empty is True when nothing qualifies.
    """
    n = len(df)
    metadata = AnalysisMetadata(
        method="fractal_swing",
        params={"lookback": lookback},
        start=df.index[0] if n else None,
        end=df.index[-1] if n else None,
        n_bars=n,
    )

    # FAILSAFE: not enough bars for a single centered swing point.
    if n < lookback * 2 + 1:
        return SwingPointResult(points=_empty_points(), metadata=metadata)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    records: list[tuple[pd.Timestamp, float, str, int]] = []
    # Center i runs from `lookback` to `n-1-lookback` inclusive — this is what
    # excludes the first/last `lookback` bars from ever being flagged.
    for i in range(lookback, n - lookback):
        win_hi = highs[i - lookback: i + lookback + 1]
        win_lo = lows[i - lookback: i + lookback + 1]

        # Strict, unique maximum of the window -> swing high.
        if highs[i] == win_hi.max() and np.count_nonzero(win_hi == highs[i]) == 1:
            records.append((df.index[i], float(highs[i]), "high", i))
        # Strict, unique minimum of the window -> swing low. (A single bar can
        # be both an outside-bar high and low; evaluating independently keeps
        # the method honest rather than silently dropping one.)
        if lows[i] == win_lo.min() and np.count_nonzero(win_lo == lows[i]) == 1:
            records.append((df.index[i], float(lows[i]), "low", i))

    if not records:
        return SwingPointResult(points=_empty_points(), metadata=metadata)

    # Keep chronological order (records are appended in ascending i, but a 'low'
    # can be appended after a 'high' at the same i — stable sort by pos fixes it).
    records.sort(key=lambda r: r[3])
    points = pd.DataFrame(
        {"price": [r[1] for r in records],
         "type": [r[2] for r in records],
         "pos": [r[3] for r in records]},
        index=pd.DatetimeIndex([r[0] for r in records], name=df.index.name),
    )
    return SwingPointResult(points=points, metadata=metadata)


# ---------------------------------------------------------------------------
# 2. Fibonacci retracement levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FibLevels:
    """
    A Fibonacci retracement grid anchored on one swing high / swing low pair.

    levels: dict {ratio -> price} for the five interior ratios in FIB_RATIOS.
            Ratio 0.0 corresponds to `top` and 1.0 to `bottom` (the anchors,
            exposed separately below).
    """
    levels: dict                       # {0.236: price, ... 0.786: price}
    swing_high: tuple                  # (timestamp, price) of the chosen high
    swing_low: tuple                   # (timestamp, price) of the chosen low
    top: float                         # higher of the two anchor prices (0%)
    bottom: float                      # lower of the two anchor prices (100%)
    direction: str                     # "up" if low precedes high, else "down"
    metadata: AnalysisMetadata


def compute_fib_levels(
    df: pd.DataFrame, swing_points: SwingPointResult
) -> Optional[FibLevels]:
    """
    Compute standard Fibonacci retracement levels from detected swing points.

    Anchor-selection rule (documented and exact):
        Use the MOST RECENT swing high and the MOST RECENT swing low (each the
        last of its type by timestamp), regardless of which occurred first, as
        long as BOTH exist. The grid is drawn between the higher and lower of
        those two prices, so the five levels are always ordered and sit inside
        the range.

    Rolling / recompute trigger (this function is stateless — the behavior
    emerges from re-calling it as new bars arrive):
        Because the anchors are always "most recent high" and "most recent low",
        the grid changes exactly when a NEWER swing high or a NEWER swing low is
        detected than the one currently anchoring it — i.e. when a new swing
        point replaces an anchor (including when it sets a new extreme). Pass
        freshly recomputed swing_points and the levels roll forward.

    Failsafes:
        * Fewer than two usable swings (need >=1 high AND >=1 low) -> None.
        * Degenerate range (|top-bottom| below _DEGENERATE_RANGE_FRAC of the
          price level, e.g. a flat market where high==low) -> None, rather than
          emitting five identical prices.

    Audit: the returned object records the exact two swing points (timestamp +
    price) used, plus metadata (including the swing lookback that produced them).
    """
    highs = swing_points.highs
    lows = swing_points.lows

    swing_lookback = swing_points.metadata.params.get("lookback")
    metadata = AnalysisMetadata(
        method="fib_retracement",
        params={"ratios": list(FIB_RATIOS), "swing_lookback": swing_lookback},
        start=df.index[0] if len(df) else None,
        end=df.index[-1] if len(df) else None,
        n_bars=len(df),
    )

    # FAILSAFE: need at least one high and one low.
    if highs.empty or lows.empty:
        return None

    high_time = highs.index[-1]
    low_time = lows.index[-1]
    high_price = float(highs.iloc[-1]["price"])
    low_price = float(lows.iloc[-1]["price"])

    top = max(high_price, low_price)
    bottom = min(high_price, low_price)
    rng = top - bottom

    # FAILSAFE: degenerate (near-zero) range.
    if rng < _DEGENERATE_RANGE_FRAC * max(abs(top), 1.0):
        return None

    levels = {ratio: top - ratio * rng for ratio in FIB_RATIOS}
    direction = "up" if low_time < high_time else "down"

    return FibLevels(
        levels=levels,
        swing_high=(high_time, high_price),
        swing_low=(low_time, low_price),
        top=top,
        bottom=bottom,
        direction=direction,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 3. Trendline (linear regression through recent swing points)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trendline:
    """
    A straight line fit through recent swing points of one type.

    Geometry is expressed in (position, price) space, where position is the
    integer bar index within the analyzed DataFrame. y = slope*pos + intercept.

    x_start/x_end are positions (x_end may be fractional and project past the
    last bar). start_time/end_time are the corresponding timestamps (end_time is
    extrapolated at the average bar interval when the projection runs past the
    final bar).
    """
    side: str                          # "support" (lows) or "resistance" (highs)
    slope: float                       # price change per bar
    intercept: float
    r_squared: float
    low_confidence: bool               # True when r_squared < LOW_CONFIDENCE_R2
    n_points: int                      # number of swing points actually fit
    points_used: pd.DataFrame          # the swing points used (subset)
    x_start: float                     # position of first contributing point
    x_end: float                       # projected end position
    y_start: float
    y_end: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    metadata: AnalysisMetadata


def fit_trendline(
    df: pd.DataFrame,
    swing_points: SwingPointResult,
    side: str = "resistance",
    min_points: int = 3,
    n_points: int = 5,
) -> Optional[Trendline]:
    """
    Fit a trendline through the most recent swing points of one type by ordinary
    least-squares linear regression (price vs. bar position).

    side:       "resistance" fits swing HIGHs; "support" fits swing LOWs.
    min_points: minimum swing points of that type required to fit at all.
    n_points:   use at most this many of the MOST RECENT qualifying swings.

    This is ONE algorithmic method among several valid ones. A human chartist
    would often anchor on two touches, weight the most recent reaction, or draw
    along wicks vs. bodies — and would frequently draw a different line. This
    regression is reproducible and auditable, not authoritative.

    Failsafes:
        * Fewer than `min_points` swings of the relevant type -> None (we do not
          fit a "trend" through one or two points).
        * Fit quality is reported honestly: R^2 is always computed, and a fit
          with R^2 < LOW_CONFIDENCE_R2 (0.5) is still returned but tagged
          low_confidence=True. The function never hides a poor fit; the renderer
          decides whether to gray it out or skip it.
        * The drawn line is bounded: it is projected only TRENDLINE_PROJECTION_FRAC
          (20%) of the swing-point span past the last contributing swing, not
          indefinitely across the chart.

    Rolling: stateless — refit by re-calling with freshly recomputed
    swing_points whenever a new swing appears.

    LOOKAHEAD: built on centered swing points; visualization-only (see module
    docstring).
    """
    if side == "resistance":
        pts = swing_points.highs
    elif side == "support":
        pts = swing_points.lows
    else:
        raise ValueError(f"side must be 'resistance' or 'support', got {side!r}")

    swing_lookback = swing_points.metadata.params.get("lookback")
    metadata = AnalysisMetadata(
        method="ols_trendline",
        params={
            "side": side,
            "min_points": min_points,
            "n_points": n_points,
            "swing_lookback": swing_lookback,
        },
        start=df.index[0] if len(df) else None,
        end=df.index[-1] if len(df) else None,
        n_bars=len(df),
    )

    # FAILSAFE: not enough swings of this type to define a trend.
    # (A line also mathematically needs >= 2 points; min_points default 3 covers
    # this, but guard explicitly in case a caller lowers min_points.)
    if len(pts) < max(min_points, 2):
        return None

    used = pts.iloc[-n_points:]  # most recent N
    x = used["pos"].to_numpy(dtype=float)
    y = used["price"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # ss_tot == 0 means the swings are perfectly horizontal — a flat support /
    # resistance, which IS a perfect (slope 0) linear fit, so R^2 = 1.0.
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    low_confidence = r_squared < LOW_CONFIDENCE_R2

    first_pos = float(x[0])
    last_pos = float(x[-1])
    span = last_pos - first_pos
    x_start = first_pos
    x_end = last_pos + TRENDLINE_PROJECTION_FRAC * span
    y_start = float(slope * x_start + intercept)
    y_end = float(slope * x_end + intercept)

    # Timestamps. start_time is the first contributing swing's actual time.
    # end_time is extrapolated at the average bar interval, since x_end may sit
    # past the final bar of df.
    start_time = used.index[0]
    n = len(df)
    if n >= 2:
        bar_interval = (df.index[-1] - df.index[0]) / (n - 1)
    else:
        bar_interval = pd.Timedelta(0)
    last_time = used.index[-1]
    end_time = last_time + (x_end - last_pos) * bar_interval

    return Trendline(
        side=side,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        low_confidence=bool(low_confidence),
        n_points=len(used),
        points_used=used,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        start_time=start_time,
        end_time=end_time,
        metadata=metadata,
    )
