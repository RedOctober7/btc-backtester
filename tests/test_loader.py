import pytest
import pandas as pd
from data.loader import _validate_df, _klines_to_df, _discard_unfinished_candles


def _make_raw_klines(prices: list[float], start_ms: int = 0, step_ms: int = 14_400_000) -> list:
    """Build fake Binance raw kline rows for testing."""
    rows = []
    for i, p in enumerate(prices):
        open_time = start_ms + i * step_ms
        close_time = open_time + step_ms - 1
        rows.append([open_time, str(p), str(p), str(p), str(p), "1.0",
                      close_time, "0", "0", "0", "0", "0"])
    return rows


# ---------------------------------------------------------------------------
# _klines_to_df
# ---------------------------------------------------------------------------

def test_klines_to_df_parses_fields():
    raw = _make_raw_klines([100.0, 200.0])
    df = _klines_to_df(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "close_time"]
    assert df["close"].iloc[0] == 100.0
    assert df["close"].iloc[1] == 200.0
    # Index must be UTC-aware
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_klines_to_df_empty_raises():
    with pytest.raises(ValueError, match="No klines returned"):
        _klines_to_df([])


# ---------------------------------------------------------------------------
# _validate_df
# ---------------------------------------------------------------------------

def test_validate_df_rejects_unsorted():
    """Candles not sorted ascending by open_time must raise ValueError."""
    raw = _make_raw_klines([100.0, 200.0, 300.0])
    df = _klines_to_df(raw)
    # Reverse the order to break sorting
    df = df.iloc[::-1]
    with pytest.raises(ValueError, match="not sorted ascending"):
        _validate_df(df)


def test_validate_df_removes_duplicates(caplog):
    """Exact duplicate open_times are removed with a warning."""
    import logging
    raw = _make_raw_klines([100.0, 100.0, 200.0])
    df = _klines_to_df(raw)
    # Force a duplicate by setting the second row's index equal to the first
    df.index = [df.index[0], df.index[0], df.index[2]]
    with caplog.at_level(logging.WARNING):
        cleaned = _validate_df(df)
    assert len(cleaned) == 2
    assert "duplicate" in caplog.text.lower()


def test_validate_df_detects_gaps(caplog):
    """Gaps larger than the expected 4h interval are logged as warnings."""
    import logging
    # 4h in ms = 14_400_000; create a gap of 8h between bar 1 and bar 2
    raw = _make_raw_klines([100.0, 200.0, 300.0])
    df = _klines_to_df(raw)
    # Move the last candle 4h further out to create a gap
    new_index = list(df.index)
    new_index[2] = new_index[2] + pd.Timedelta("4h")
    df.index = pd.DatetimeIndex(new_index, tz="UTC")
    with caplog.at_level(logging.WARNING):
        _validate_df(df)
    assert "gap" in caplog.text.lower()


# ---------------------------------------------------------------------------
# _discard_unfinished_candles
# ---------------------------------------------------------------------------

def test_discard_unfinished_candles_drops_future_close():
    """Candles whose close_time is in the future must be dropped."""
    now = pd.Timestamp.now(tz="UTC")
    dates = pd.date_range("2020-01-01", periods=2, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": [100.0, 200.0], "high": [110.0, 210.0],
         "low": [90.0, 190.0], "close": [105.0, 205.0], "volume": [1.0, 1.0],
         "close_time": [dates[0] + pd.Timedelta("4h"), now + pd.Timedelta("2h")]},
        index=dates,
    )
    cleaned = _discard_unfinished_candles(df)
    assert len(cleaned) == 1
    assert cleaned.index[0] == dates[0]
