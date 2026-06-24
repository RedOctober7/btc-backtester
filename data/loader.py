"""
Binance 4h klines fetcher with local parquet cache.

Design notes:
- Candles are uniquely identified by open_time (per Binance docs); index is always open_time UTC.
- Primary host: data-api.binance.vision — public data mirror, no API key, no trading endpoints.
  Fallback: api.binance.com — hits region blocks more often but kept as backup.
- Cache key = symbol + interval + start_date + end_date; re-fetch only when range changes.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL_PRIMARY = "https://data-api.binance.vision"
BASE_URL_FALLBACK = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_KLINES_PER_REQUEST = 1000  # Binance hard cap per response

# Kline array field positions (per Binance REST API docs — order is guaranteed)
_COL_OPEN_TIME = 0
_COL_OPEN = 1
_COL_HIGH = 2
_COL_LOW = 3
_COL_CLOSE = 4
_COL_VOLUME = 5
_COL_CLOSE_TIME = 6

CACHE_DIR = Path("cache")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _klines_to_df(raw: list[list]) -> pd.DataFrame:
    """Convert raw Binance kline array to a DataFrame indexed by open_time (UTC)."""
    if not raw:
        raise ValueError("No klines returned")

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    # Parse timestamps from milliseconds to UTC-aware Timestamps
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Index by open_time — the canonical identifier for a candle
    df = df.set_index("open_time")
    return df[["open", "high", "low", "close", "volume", "close_time"]]


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce data hygiene: sorted ascending, de-duplicated, gaps reported.
    Fails loudly on malformed input rather than silently guessing.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Candles are not sorted ascending by open_time — data is malformed; "
            "sort before passing to the backtester"
        )

    # De-duplicate by open_time (keep first occurrence)
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_removed = n_before - len(df)
    if n_removed:
        logger.warning("Removed %d duplicate candle(s) by open_time", n_removed)

    # Detect gaps: 4h bars should be spaced exactly 4h apart
    expected_delta = pd.Timedelta("4h")
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > expected_delta]
    if not gaps.empty:
        logger.warning("Detected %d gap(s) in candle data (Binance outage / missing bars):", len(gaps))
        for ts, gap in gaps.items():
            logger.warning("  Gap of %s ending at %s", gap, ts)

    return df


def _discard_unfinished_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop candles whose close_time is in the future.
    Binance returns the currently-forming candle when the range ends near 'now';
    its OHLC changes every tick, so backtesting on it would give different results
    each run. Only fully-closed candles are allowed.
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    mask = df["close_time"] <= now_utc
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.debug("Dropped %d unfinished candle(s) (close_time in the future)", n_dropped)
    return df[mask]


def _fetch_page(
    session: requests.Session,
    base_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_KLINES_PER_REQUEST,
    }
    resp = session.get(f"{base_url}{KLINES_ENDPOINT}", params=params, timeout=30)

    if resp.status_code in (429, 418):
        # 429 = rate-limited; 418 = IP auto-banned; both require backing off
        raise _RateLimitError(f"HTTP {resp.status_code}")
    if resp.status_code in (403, 451):
        # Firewall or region block — the host itself is unreachable for us
        raise _RegionBlockError(f"HTTP {resp.status_code} from {base_url}")

    resp.raise_for_status()
    return resp.json()


class _RateLimitError(Exception):
    pass


class _RegionBlockError(Exception):
    pass


def _fetch_all_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    """Paginate through all klines; handle rate limits and host fallback."""
    session = requests.Session()
    all_klines: list[list] = []
    current_start = start_ms
    base_url = BASE_URL_PRIMARY
    backoff = 1  # seconds; doubles on each rate-limit hit

    while current_start < end_ms:
        try:
            page = _fetch_page(session, base_url, symbol, interval, current_start, end_ms)
            backoff = 1  # reset on success
        except _RateLimitError as e:
            logger.warning("Rate limited (%s); sleeping %ds before retry", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        except _RegionBlockError:
            if base_url == BASE_URL_PRIMARY:
                logger.warning(
                    "Primary host %s is blocked; switching to fallback %s",
                    BASE_URL_PRIMARY,
                    BASE_URL_FALLBACK,
                )
                base_url = BASE_URL_FALLBACK
                continue
            raise RuntimeError(
                f"Both Binance hosts are unreachable. "
                f"Check your network or VPN and retry."
            )

        if not page:
            break

        all_klines.extend(page)

        last_open_ms = page[-1][_COL_OPEN_TIME]
        # Advance past the last returned candle's open_time to avoid re-fetching it
        current_start = last_open_ms + 1

        if len(page) < MAX_KLINES_PER_REQUEST:
            break  # last page received

    return all_klines


def _cache_path(symbol: str, interval: str, start_date: str, end_date: str) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}_{start_date}_{end_date}.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_candles(
    symbol: str = "BTCUSDT",
    interval: str = "4h",
    start_date: str = "2022-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV candles. Returns from local parquet cache when available;
    fetches from Binance and caches on the first call for a given range.

    Returns a DataFrame indexed by open_time (UTC) with columns:
        open, high, low, close, volume
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache_file = _cache_path(symbol, interval, start_date, end_date)

    if cache_file.exists():
        logger.info("Loading candles from cache: %s", cache_file)
        df = pd.read_parquet(cache_file)
        if df.index.tz is None:
            # Parquet may strip timezone info; restore it
            df.index = df.index.tz_localize("UTC")
        return df

    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)

    logger.info("Fetching %s %s candles from Binance (%s -> %s)...", symbol, interval, start_date, end_date)
    raw = _fetch_all_klines(symbol, interval, start_ms, end_ms)

    df = _klines_to_df(raw)
    df = _discard_unfinished_candles(df)
    df = _validate_df(df)
    df = df.drop(columns=["close_time"])  # only needed for the unfinished-candle check

    CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache_file)
    logger.info("Cached %d candles to %s", len(df), cache_file)

    return df
