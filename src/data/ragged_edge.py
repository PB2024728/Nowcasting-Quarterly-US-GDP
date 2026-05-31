"""
Ragged-edge masking: simulate what a forecaster would have seen on a given as-of date.

For each (period, series) cell in a monthly panel, the cell is set to NaN if
the underlying release would not yet have been published on `as_of`:

    release_date = period.end_time.date() + timedelta(days=lag)
    if as_of < release_date  →  NaN

Publication lags (days after reference month-end) come from config.PUBLICATION_LAGS_DAYS.
Series not in that dict default to a 30-day lag.

Public API:
    mask_ragged_edge(panel, as_of) -> pd.DataFrame
    vintage_as_of(quarter, vintage) -> date
        Helper that converts a (quarter, vintage-number) pair to a concrete as-of date.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import PUBLICATION_LAGS_DAYS

_DEFAULT_LAG = 30  # days, used for any series not in PUBLICATION_LAGS_DAYS


def mask_ragged_edge(panel: pd.DataFrame, as_of: date | str | pd.Timestamp) -> pd.DataFrame:
    """
    Return a copy of `panel` with cells masked to NaN where the data would not
    yet have been published on `as_of`.

    Parameters
    ----------
    panel : DataFrame with PeriodIndex(freq='M')
    as_of : the date the forecaster is standing on

    Returns
    -------
    DataFrame of the same shape; unreleased cells are NaN.
    """
    if isinstance(as_of, str):
        as_of = pd.Timestamp(as_of).date()
    elif isinstance(as_of, pd.Timestamp):
        as_of = as_of.date()

    panel = panel.copy()

    # Vectorised: build one array of period-end dates, then mask column by column
    period_end_dates = np.array([p.end_time.date() for p in panel.index])

    for col in panel.columns:
        lag = PUBLICATION_LAGS_DAYS.get(col, _DEFAULT_LAG)
        release_dates = np.array([d + timedelta(days=lag) for d in period_end_dates])
        not_yet_released = np.array([as_of < rd for rd in release_dates])
        panel.loc[not_yet_released, col] = np.nan

    return panel


def vintage_as_of(quarter: str | pd.Period, vintage: int) -> date:
    """
    Return the as-of date for a given quarter and within-quarter vintage number.

    vintage=1 → last day of the first month of the quarter
    vintage=2 → last day of the second month
    vintage=3 → last day of the third (final) month of the quarter

    Example: vintage_as_of('2010Q1', 2) → date(2010, 2, 28)
    """
    if not isinstance(quarter, pd.Period):
        quarter = pd.Period(quarter, freq="Q")
    if vintage not in (1, 2, 3):
        raise ValueError(f"vintage must be 1, 2, or 3; got {vintage}")

    # Quarter start month (e.g., Q1 → Jan=1, Q2 → Apr=4, ...)
    qstart_month = (quarter.quarter - 1) * 3 + 1
    target_month = qstart_month + (vintage - 1)
    target_period = pd.Period(year=quarter.year, month=target_month, freq="M")
    # Last calendar day of that month
    return target_period.end_time.date()
