import pandas as pd


def make_ohlcv(rows: list[tuple[float, float, float, float]], freq: str = "4h") -> pd.DataFrame:
    """
    Helper used by tests to build a synthetic OHLCV DataFrame.
    rows: list of (open, high, low, close) — volume defaults to 1.0.
    Returns a DataFrame indexed by UTC open_time at 4h intervals.
    """
    n = len(rows)
    dates = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    opens, highs, lows, closes = zip(*rows)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1.0] * n},
        index=dates,
    )
