"""
Unit tests for src.data.ragged_edge.mask_ragged_edge.

Reference period: March 2024 (ends 2024-03-31).
Release dates for each series (period-end + lag from config):
    UMCSENT  (lag=0)  → 2024-03-31
    T10Y2Y   (lag=1)  → 2024-04-01
    BAA10Y   (lag=1)  → 2024-04-01
    NASDAQCOM(lag=1)  → 2024-04-01
    PAYEMS   (lag=5)  → 2024-04-05
    ICSA     (lag=5)  → 2024-04-05
    UNRATE   (lag=5)  → 2024-04-05
    RSAFS    (lag=14) → 2024-04-14
    INDPRO   (lag=17) → 2024-04-17
    HOUST    (lag=19) → 2024-04-19
    DGORDER  (lag=28) → 2024-04-28
    PCEPI    (lag=28) → 2024-04-28
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import PUBLICATION_LAGS_DAYS
from src.data.ragged_edge import mask_ragged_edge, vintage_as_of


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(periods: list[str]) -> pd.DataFrame:
    """Synthetic panel with all series in PUBLICATION_LAGS_DAYS; all values = 1.0."""
    idx = pd.PeriodIndex(periods, freq="M")
    data = {col: np.ones(len(idx)) for col in PUBLICATION_LAGS_DAYS}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Test 1: Early in month — most-recent month still largely hidden
# ---------------------------------------------------------------------------

def test_early_in_month_hides_slow_series():
    """
    as_of = 2024-04-03 (3 days into April).

    Expected:
      - March PAYEMS (release 2024-04-05) → NaN
      - March UMCSENT (release 2024-03-31) → visible (lag=0, released by month-end)
      - March T10Y2Y (release 2024-04-01) → visible (1-day lag, 04-01 <= 04-03)
      - February PAYEMS (release 2024-03-05) → visible (already out)
    """
    panel = _make_panel(["2024-02", "2024-03"])
    masked = mask_ragged_edge(panel, date(2024, 4, 3))

    # March slow-release series → hidden
    assert np.isnan(masked.loc["2024-03", "PAYEMS"]), "March PAYEMS should be NaN"
    assert np.isnan(masked.loc["2024-03", "ICSA"]),   "March ICSA should be NaN"
    assert np.isnan(masked.loc["2024-03", "INDPRO"]), "March INDPRO should be NaN"

    # March fast-release series → visible
    assert not np.isnan(masked.loc["2024-03", "UMCSENT"]),   "March UMCSENT should be visible"
    assert not np.isnan(masked.loc["2024-03", "T10Y2Y"]),    "March T10Y2Y should be visible"
    assert not np.isnan(masked.loc["2024-03", "NASDAQCOM"]), "March NASDAQCOM should be visible"

    # February data → fully visible regardless of series
    assert not np.isnan(masked.loc["2024-02", "PAYEMS"]), "Feb PAYEMS should be visible"
    assert not np.isnan(masked.loc["2024-02", "DGORDER"]), "Feb DGORDER should be visible"


# ---------------------------------------------------------------------------
# Test 2: Mid-month — split between released and unreleased
# ---------------------------------------------------------------------------

def test_mid_month_split():
    """
    as_of = 2024-04-15.

    Expected for March 2024:
      - RSAFS   (release 2024-04-14) → visible (14 <= 15)
      - INDPRO  (release 2024-04-17) → NaN (17 > 15)
      - HOUST   (release 2024-04-19) → NaN (19 > 15)
      - DGORDER (release 2024-04-28) → NaN (28 > 15)
    """
    panel = _make_panel(["2024-03"])
    masked = mask_ragged_edge(panel, date(2024, 4, 15))

    assert not np.isnan(masked.loc["2024-03", "RSAFS"]),   "RSAFS should be visible"
    assert not np.isnan(masked.loc["2024-03", "PAYEMS"]),  "PAYEMS should be visible"
    assert not np.isnan(masked.loc["2024-03", "UNRATE"]),  "UNRATE should be visible"

    assert np.isnan(masked.loc["2024-03", "INDPRO"]),  "INDPRO should be NaN"
    assert np.isnan(masked.loc["2024-03", "HOUST"]),   "HOUST should be NaN"
    assert np.isnan(masked.loc["2024-03", "DGORDER"]), "DGORDER should be NaN"
    assert np.isnan(masked.loc["2024-03", "PCEPI"]),   "PCEPI should be NaN"


# ---------------------------------------------------------------------------
# Test 3: Well after the period — everything visible
# ---------------------------------------------------------------------------

def test_well_after_period_all_visible():
    """
    as_of = 2024-05-31 (two months after March).
    All March observations must be visible; no NaN anywhere.
    """
    panel = _make_panel(["2024-03"])
    masked = mask_ragged_edge(panel, date(2024, 5, 31))

    for col in panel.columns:
        assert not np.isnan(masked.loc["2024-03", col]), f"{col} should be visible on 2024-05-31"


# ---------------------------------------------------------------------------
# Bonus: vintage_as_of helper
# ---------------------------------------------------------------------------

def test_vintage_as_of_q1():
    assert vintage_as_of("2010Q1", 1) == date(2010, 1, 31)
    assert vintage_as_of("2010Q1", 2) == date(2010, 2, 28)
    assert vintage_as_of("2010Q1", 3) == date(2010, 3, 31)


def test_vintage_as_of_q4():
    assert vintage_as_of("2023Q4", 1) == date(2023, 10, 31)
    assert vintage_as_of("2023Q4", 3) == date(2023, 12, 31)
