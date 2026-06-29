"""
Tests for the standalone technical-analysis module (analysis/levels.py).

Every numeric expectation here is hand-computed, the same discipline used in
test_metrics.py. Synthetic series have known, hand-placed extrema so we can
assert the detector finds EXACTLY those points and nothing at the edges.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.levels import (
    find_swing_points,
    compute_fib_levels,
    fit_trendline,
    SwingPointResult,
    AnalysisMetadata,
    FIB_RATIOS,
    LOW_CONFIDENCE_R2,
    TRENDLINE_PROJECTION_FRAC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_from_base(base: list[float], wick: float = 0.1) -> pd.DataFrame:
    """
    Build an OHLCV frame whose high = base+wick and low = base-wick, so swing
    highs land on local maxima of `base` and swing lows on local minima.
    """
    idx = pd.date_range("2024-01-01", periods=len(base), freq="4h", tz="UTC")
    b = np.asarray(base, dtype=float)
    return pd.DataFrame(
        {"open": b, "high": b + wick, "low": b - wick, "close": b,
         "volume": np.ones(len(base))},
        index=idx,
    )


def _make_swings(specs: list[tuple[int, float, str]], n_bars: int = 64,
                 lookback: int = 5) -> tuple[SwingPointResult, pd.DataFrame]:
    """
    Construct a SwingPointResult directly from (pos, price, type) specs, so the
    fib/trendline tests can exercise exact prices without depending on the
    detector. Timestamps are derived from a regular 4h grid.
    """
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="4h", tz="UTC")
    specs = sorted(specs, key=lambda s: s[0])
    points = pd.DataFrame(
        {"price": [s[1] for s in specs],
         "type": [s[2] for s in specs],
         "pos": [s[0] for s in specs]},
        index=pd.DatetimeIndex([idx[s[0]] for s in specs], name=idx.name),
    )
    meta = AnalysisMetadata(method="fractal_swing", params={"lookback": lookback},
                            start=idx[0], end=idx[-1], n_bars=n_bars)
    return SwingPointResult(points=points, metadata=meta), _grid_df(idx)


def _grid_df(idx: pd.DatetimeIndex) -> pd.DataFrame:
    flat = np.ones(len(idx))
    return pd.DataFrame({"open": flat, "high": flat, "low": flat, "close": flat,
                         "volume": flat}, index=idx)


# ---------------------------------------------------------------------------
# 1. Swing-point detection
# ---------------------------------------------------------------------------

def test_swing_points_found_at_exact_hand_placed_extrema():
    # base local maxima at pos 2 and 7; local minima at pos 5 and 10.
    base = [10, 12, 15, 12, 10, 8, 11, 14, 11, 9, 7, 9, 12]
    df = _df_from_base(base)
    result = find_swing_points(df, lookback=2)

    assert list(result.highs["pos"]) == [2, 7]
    assert list(result.lows["pos"]) == [5, 10]
    # Prices reflect the wick-extended high/low, not the base.
    assert result.highs.iloc[0]["price"] == pytest.approx(15 + 0.1)
    assert result.lows.iloc[0]["price"] == pytest.approx(8 - 0.1)
    # Metadata is tagged with the lookback used.
    assert result.metadata.params["lookback"] == 2
    assert result.metadata.method == "fractal_swing"


def test_swing_points_never_flag_edges():
    # A monotonic ramp has its max at the very last bar and min at the first —
    # both are edges and must NOT be reported as swings.
    base = list(range(20))
    df = _df_from_base(base)
    result = find_swing_points(df, lookback=3)
    assert result.is_empty
    # And the reverse ramp (max at first bar).
    df_rev = _df_from_base(list(range(20))[::-1])
    assert find_swing_points(df_rev, lookback=3).is_empty


def test_swing_points_empty_when_series_too_short():
    # lookback*2+1 = 11 bars required; 10 bars must return empty, not crash.
    base = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
    df = _df_from_base(base)
    result = find_swing_points(df, lookback=5)
    assert result.is_empty
    assert result.metadata.n_bars == 10
    # Boundary: exactly 11 bars is enough to evaluate the single center bar.
    base11 = [1, 2, 3, 4, 5, 9, 5, 4, 3, 2, 1]
    df11 = _df_from_base(base11)
    res11 = find_swing_points(df11, lookback=5)
    assert list(res11.highs["pos"]) == [5]


def test_swing_points_strict_plateau_not_flagged():
    # A flat top (tie for the window max) is not a unique extreme -> no swing.
    base = [1, 2, 5, 5, 2, 1, 2, 1, 0, 1, 2]
    df = _df_from_base(base, wick=0.0)
    result = find_swing_points(df, lookback=2)
    # The plateau at pos 2-3 must not produce a swing high.
    assert 2 not in list(result.highs["pos"])
    assert 3 not in list(result.highs["pos"])


# ---------------------------------------------------------------------------
# 2. Fibonacci levels
# ---------------------------------------------------------------------------

def test_fib_levels_hand_computed():
    # Most recent high = 100 (pos 30), most recent low = 50 (pos 20).
    # top=100, bottom=50, range=50.
    swings, df = _make_swings([(20, 50.0, "low"), (30, 100.0, "high")])
    fib = compute_fib_levels(df, swings)
    assert fib is not None

    expected = {
        0.236: 100 - 0.236 * 50,  # 88.2
        0.382: 100 - 0.382 * 50,  # 80.9
        0.5:   100 - 0.5 * 50,    # 75.0
        0.618: 100 - 0.618 * 50,  # 69.1
        0.786: 100 - 0.786 * 50,  # 60.7
    }
    for ratio in FIB_RATIOS:
        assert fib.levels[ratio] == pytest.approx(expected[ratio], abs=1e-9)

    # Anchors and audit fields.
    assert fib.top == pytest.approx(100.0)
    assert fib.bottom == pytest.approx(50.0)
    assert fib.swing_high[1] == pytest.approx(100.0)
    assert fib.swing_low[1] == pytest.approx(50.0)
    # low (pos 20) precedes high (pos 30) -> retracement of an up-move.
    assert fib.direction == "up"
    assert fib.metadata.params["swing_lookback"] == 5


def test_fib_uses_most_recent_pair():
    # Two highs and two lows; the MOST RECENT of each must anchor the grid.
    swings, df = _make_swings([
        (5, 80.0, "high"), (10, 40.0, "low"),
        (25, 120.0, "high"), (30, 60.0, "low"),
    ])
    fib = compute_fib_levels(df, swings)
    assert fib is not None
    assert fib.top == pytest.approx(120.0)     # most recent high
    assert fib.bottom == pytest.approx(60.0)   # most recent low
    assert fib.swing_high[1] == pytest.approx(120.0)
    assert fib.swing_low[1] == pytest.approx(60.0)


def test_fib_returns_none_on_degenerate_flat_range():
    # Swing high price == swing low price -> zero range -> None.
    swings, df = _make_swings([(20, 100.0, "low"), (30, 100.0, "high")])
    assert compute_fib_levels(df, swings) is None


def test_fib_returns_none_when_missing_a_side():
    # Only highs, no lows -> cannot anchor -> None.
    swings, df = _make_swings([(10, 100.0, "high"), (20, 110.0, "high")])
    assert compute_fib_levels(df, swings) is None


# ---------------------------------------------------------------------------
# 3. Trendline
# ---------------------------------------------------------------------------

def test_trendline_recovers_known_line():
    # Resistance swings sit exactly on y = 2*pos + 10.
    specs = [(p, 2.0 * p + 10.0, "high") for p in (0, 5, 10, 15, 20)]
    swings, df = _make_swings(specs, n_bars=40)
    tl = fit_trendline(df, swings, side="resistance", min_points=3)

    assert tl is not None
    assert tl.slope == pytest.approx(2.0, abs=1e-9)
    assert tl.intercept == pytest.approx(10.0, abs=1e-9)
    assert tl.r_squared == pytest.approx(1.0, abs=1e-9)
    assert tl.low_confidence is False
    assert tl.side == "resistance"
    assert tl.n_points == 5


def test_trendline_projection_bound_is_20_percent():
    # span = 20 - 0 = 20; x_end must be last_pos + 0.20*span = 20 + 4 = 24.
    specs = [(p, 2.0 * p + 10.0, "high") for p in (0, 5, 10, 15, 20)]
    swings, df = _make_swings(specs, n_bars=64)
    tl = fit_trendline(df, swings, side="resistance", min_points=3)
    assert tl.x_start == pytest.approx(0.0)
    assert tl.x_end == pytest.approx(20 + TRENDLINE_PROJECTION_FRAC * 20)  # 24.0


def test_trendline_min_points_failsafe():
    # Only two highs but min_points=3 -> None (don't fit a "trend" through 2).
    specs = [(5, 30.0, "high"), (10, 40.0, "high")]
    swings, df = _make_swings(specs)
    assert fit_trendline(df, swings, side="resistance", min_points=3) is None


def test_trendline_low_confidence_flag_on_scatter():
    # Deliberately scattered highs -> poor linear fit -> R^2 < 0.5 tagged.
    specs = [(0, 50.0, "high"), (5, 10.0, "high"), (10, 55.0, "high"),
             (15, 12.0, "high"), (20, 48.0, "high")]
    swings, df = _make_swings(specs, n_bars=40)
    tl = fit_trendline(df, swings, side="resistance", min_points=3)
    assert tl is not None                       # still returned, not hidden
    assert tl.r_squared < LOW_CONFIDENCE_R2
    assert tl.low_confidence is True


def test_trendline_support_uses_lows():
    specs = [(p, -1.5 * p + 200.0, "low") for p in (2, 8, 14, 20, 26)]
    swings, df = _make_swings(specs, n_bars=40)
    tl = fit_trendline(df, swings, side="support", min_points=3)
    assert tl is not None
    assert tl.side == "support"
    assert tl.slope == pytest.approx(-1.5, abs=1e-9)
    assert tl.r_squared == pytest.approx(1.0, abs=1e-9)


def test_trendline_invalid_side_raises():
    swings, df = _make_swings([(p, float(p), "high") for p in (0, 5, 10, 15)])
    with pytest.raises(ValueError):
        fit_trendline(df, swings, side="diagonal")


# ---------------------------------------------------------------------------
# 4. Decoupling — analysis must not leak into engine/strategies/broker
# ---------------------------------------------------------------------------

def _imported_roots(py_path: Path) -> set[str]:
    """
    Return the set of top-level module names actually imported by a .py file,
    parsed from its AST (so prose in docstrings/comments never registers as an
    import). For `from engine.broker import X`, the root is 'engine'.
    """
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


def test_analysis_does_not_leak_into_engine_or_strategies():
    """
    The engine, broker, and strategies must never import the analysis package.
    This keeps the technical-analysis layer a pure consumer-facing add-on with
    no path into trading logic (where its centered/lookahead method would be
    unsafe).
    """
    project_root = Path(__file__).resolve().parents[1]
    offenders = []
    for d in ("engine", "strategies"):
        for py in (project_root / d).rglob("*.py"):
            if "analysis" in _imported_roots(py):
                offenders.append(str(py.relative_to(project_root)))
    assert offenders == [], f"analysis leaked into trading layer: {offenders}"


def test_analysis_module_imports_no_engine_code():
    """analysis/levels.py itself must not import engine/strategies/broker."""
    project_root = Path(__file__).resolve().parents[1]
    roots = _imported_roots(project_root / "analysis" / "levels.py")
    forbidden = roots & {"engine", "strategies", "broker"}
    assert not forbidden, f"levels.py must not import {forbidden}"
