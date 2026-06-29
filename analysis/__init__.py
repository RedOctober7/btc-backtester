"""
Standalone technical-analysis primitives (swing points, Fibonacci levels,
trendlines).

This package is intentionally decoupled from the backtest engine: it depends
only on pandas and numpy, and imports nothing from engine/, strategies/, or
broker. Both run.py and a future UI can call these functions identically and
get back plain structured data (dataclasses) — never chart objects.
"""
from analysis.levels import (
    AnalysisMetadata,
    SwingPointResult,
    FibLevels,
    Trendline,
    find_swing_points,
    compute_fib_levels,
    fit_trendline,
    FIB_RATIOS,
    LOW_CONFIDENCE_R2,
    TRENDLINE_PROJECTION_FRAC,
)

__all__ = [
    "AnalysisMetadata",
    "SwingPointResult",
    "FibLevels",
    "Trendline",
    "find_swing_points",
    "compute_fib_levels",
    "fit_trendline",
    "FIB_RATIOS",
    "LOW_CONFIDENCE_R2",
    "TRENDLINE_PROJECTION_FRAC",
]
