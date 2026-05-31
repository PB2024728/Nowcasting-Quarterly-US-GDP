"""
Stationarity transforms and panel construction.

tcode conventions (following FRED-MD):
  1 = first difference
  5 = log first difference  (100 * log(x_t / x_{t-1}))

GDP target: 400 * log(GDP_t / GDP_{t-1})  — annualized QoQ log growth

Usage (module mode):
    python -m src.data.transforms       # build and save processed panels
    python -m src.data.transforms --refresh  # re-download first
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    GDP_SERIES,
    PREDICTOR_SERIES,
)
from src.data.fetch import load_raw


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def apply_tcode(s: pd.Series, tcode: int) -> pd.Series:
    """Apply a FRED-MD tcode transform to a Series; returns same-length Series with leading NaN."""
    if tcode == 1:
        return s.diff()
    if tcode == 5:
        # Guard against non-positive values before log (shouldn't occur for these series)
        with np.errstate(invalid="ignore"):
            return np.log(s.astype(float)).diff()
    raise ValueError(f"Unsupported tcode {tcode}. Supported: 1, 5.")


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def build_monthly_panel() -> pd.DataFrame:
    """
    Load all monthly predictors from cache, apply tcode transforms, and
    return an aligned DataFrame with PeriodIndex(freq='M').

    NaNs are preserved explicitly:
      - First observation of each series (diff requires a lag)
      - RSAFS pre-1992 (series unavailable before then)
    """
    frames: dict[str, pd.Series] = {}
    for sid, meta in PREDICTOR_SERIES.items():
        raw = load_raw(sid)
        transformed = apply_tcode(raw, meta["tcode"])
        transformed.name = sid
        frames[sid] = transformed

    # pd.DataFrame aligns on union of all PeriodIndices, filling gaps with NaN
    panel = pd.DataFrame(frames)
    panel.index = pd.PeriodIndex(panel.index, freq="M")
    panel.index.name = "period"
    return panel


def build_quarterly_target() -> pd.Series:
    """
    Return annualized quarter-on-quarter log growth of real GDP.
    Formula: 400 * log(GDP_t / GDP_{t-1})
    """
    gdp = load_raw(GDP_SERIES).astype(float)
    growth = 400.0 * np.log(gdp / gdp.shift(1))
    growth.name = "gdp_growth"
    growth.index = pd.PeriodIndex(growth.index, freq="Q")
    growth.index.name = "period"
    return growth


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_panels(monthly_panel: pd.DataFrame, quarterly_target: pd.Series) -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mp = monthly_panel.copy()
    mp.index = mp.index.astype(str)
    mp.to_parquet(DATA_PROCESSED_DIR / "monthly_panel.parquet")

    qt = quarterly_target.to_frame()
    qt.index = qt.index.astype(str)
    qt.to_parquet(DATA_PROCESSED_DIR / "quarterly_gdp.parquet")


def load_monthly_panel() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "monthly_panel.parquet")
    df.index = pd.PeriodIndex(df.index, freq="M")
    df.index.name = "period"
    return df


def load_quarterly_target() -> pd.Series:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "quarterly_gdp.parquet")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    df.index.name = "period"
    return df.iloc[:, 0].rename("gdp_growth")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build transformed data panels.")
    parser.parse_args()  # no flags yet; reserved for future --refresh

    print("Building monthly panel ...")
    panel = build_monthly_panel()
    print(f"  Shape: {panel.shape}  ({panel.index[0]} – {panel.index[-1]})")
    print(f"  NaN counts:\n{panel.isna().sum().to_string()}")

    print("\nBuilding quarterly GDP growth ...")
    target = build_quarterly_target()
    print(f"  Shape: {target.shape}  ({target.index[0]} – {target.index[-1]})")
    print(f"  Mean: {target.mean():.2f}%  Std: {target.std():.2f}%")

    save_panels(panel, target)
    print(f"\nSaved to {DATA_PROCESSED_DIR}")
