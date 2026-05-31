"""
FRED data ingestion with parquet caching.

Usage (module mode):
    python -m src.data.fetch            # pull all series, use cache if present
    python -m src.data.fetch --refresh  # force re-download everything

Public API:
    fetch_series(series_id, refresh=False) -> pd.Series
    fetch_all(refresh=False)             -> dict[str, pd.Series]
    load_raw(series_id)                  -> pd.Series  (cache-only, no network)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from tqdm import tqdm

from src.config import (
    ALL_SERIES,
    DATA_RAW_DIR,
    GDP_SERIES,
    PREDICTOR_SERIES,
    SAMPLE_START,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(series_id: str) -> Path:
    return DATA_RAW_DIR / f"{series_id}.parquet"


def _get_fred() -> Fred:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Fred(api_key=key)


def _agg_to_monthly(s: pd.Series, series_id: str) -> pd.Series:
    """Aggregate daily or weekly series to a monthly PeriodIndex series."""
    # Convert DatetimeIndex -> PeriodIndex(M) then group
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index)
    monthly_period = s.index.to_period("M")

    if series_id == "ICSA":
        # Weekly initial claims: sum over the month (total claims filed)
        agg = s.groupby(monthly_period).sum()
    else:
        # Financial series (T10Y2Y, BAA10Y, SP500): monthly average
        agg = s.groupby(monthly_period).mean()

    agg.index = pd.PeriodIndex(agg.index, freq="M")
    agg.name = series_id
    return agg


def _to_monthly_period(s: pd.Series, series_id: str) -> pd.Series:
    """Ensure a series has a monthly PeriodIndex regardless of original frequency."""
    meta = PREDICTOR_SERIES.get(series_id, {})
    freq = meta.get("freq", "M")

    if freq in ("D", "W"):
        return _agg_to_monthly(s, series_id)

    # Already monthly — just re-index to PeriodIndex
    s = s.copy()
    if not isinstance(s.index, pd.PeriodIndex):
        s.index = pd.DatetimeIndex(s.index).to_period("M")
    else:
        s.index = s.index.asfreq("M")
    s.name = series_id
    return s


def _to_quarterly_period(s: pd.Series) -> pd.Series:
    """Convert the GDP series to a quarterly PeriodIndex."""
    s = s.copy()
    if not isinstance(s.index, pd.PeriodIndex):
        s.index = pd.DatetimeIndex(s.index).to_period("Q")
    else:
        s.index = s.index.asfreq("Q")
    s.name = GDP_SERIES
    return s


# ---------------------------------------------------------------------------
# Core fetch / cache functions
# ---------------------------------------------------------------------------

def fetch_series(series_id: str, refresh: bool = False) -> pd.Series:
    """
    Return a series from cache (parquet) or download from FRED if missing/refresh.
    Monthly predictors come back with PeriodIndex(freq='M').
    GDP comes back with PeriodIndex(freq='Q').
    """
    cache = _cache_path(series_id)

    if cache.exists() and not refresh:
        s = pd.read_parquet(cache).iloc[:, 0]
        freq = "Q" if series_id == GDP_SERIES else "M"
        s.index = pd.PeriodIndex(s.index, freq=freq)
        s.name = series_id
        return s

    fred = _get_fred()
    # Retry with backoff to handle FRED's rate limit (roughly 120 req/min free tier)
    for attempt in range(5):
        try:
            raw = fred.get_series(series_id, observation_start=SAMPLE_START)
            break
        except ValueError as exc:
            if "Too Many Requests" in str(exc) and attempt < 4:
                wait = 2 ** attempt * 3  # 3, 6, 12, 24 seconds
                time.sleep(wait)
            else:
                raise
    raw.name = series_id

    if series_id == GDP_SERIES:
        processed = _to_quarterly_period(raw)
    else:
        processed = _to_monthly_period(raw, series_id)

    # Persist: PeriodIndex -> string for parquet compatibility
    df = processed.to_frame()
    df.index = df.index.astype(str)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    # Polite delay after every live download: FRED free tier ~120 req/min
    time.sleep(0.6)

    return processed


def load_raw(series_id: str) -> pd.Series:
    """Load from cache only; raises FileNotFoundError if not cached."""
    cache = _cache_path(series_id)
    if not cache.exists():
        raise FileNotFoundError(f"No cache for {series_id}. Run fetch_series first.")
    s = pd.read_parquet(cache).iloc[:, 0]
    freq = "Q" if series_id == GDP_SERIES else "M"
    s.index = pd.PeriodIndex(s.index, freq=freq)
    s.name = series_id
    return s


def fetch_all(refresh: bool = False) -> dict[str, pd.Series]:
    """Fetch (or load cached) all series defined in config. Returns dict keyed by series ID."""
    results: dict[str, pd.Series] = {}
    for sid in tqdm(ALL_SERIES, desc="Fetching FRED series"):
        results[sid] = fetch_series(sid, refresh=refresh)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache FRED series.")
    parser.add_argument(
        "--refresh", action="store_true", help="Force re-download even if cache exists."
    )
    args = parser.parse_args()

    print(f"Fetching {len(ALL_SERIES)} series (refresh={args.refresh}) ...")
    data = fetch_all(refresh=args.refresh)

    print("\nSeries summary:")
    print(f"{'ID':<12} {'Freq':<6} {'Start':<12} {'End':<12} {'N obs':>7}")
    print("-" * 52)
    for sid, s in data.items():
        freq = "Q" if sid == GDP_SERIES else "M"
        print(f"{sid:<12} {freq:<6} {str(s.index[0]):<12} {str(s.index[-1]):<12} {len(s):>7}")

    print(f"\nAll cached to {DATA_RAW_DIR}")
