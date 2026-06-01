"""
Expanding-window out-of-sample evaluation loop.

For each target quarter from OOS_START onward the loop:
  1. Slices the training target to all quarters strictly before the target
  2. Optionally slices the monthly panel to the ragged-edge vintage for that target
  3. Calls forecast_fn and records (forecast, realized, error)

Public API:
    expanding_window_oos(target, forecast_fn, oos_start, monthly_panel, vintage)
        -> pd.DataFrame[forecast, realized, error]
    compute_metrics(forecasts_df)
        -> dict with rmse, mae, bias, n_quarters
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.config import COVID_END, COVID_START, OOS_START
from src.data.ragged_edge import mask_ragged_edge, vintage_as_of


def expanding_window_oos(
    target: pd.Series,
    forecast_fn: Callable,
    oos_start: str = OOS_START,
    monthly_panel: pd.DataFrame | None = None,
    vintage: int | None = None,
) -> pd.DataFrame:
    """
    Expanding-window OOS loop.

    Parameters
    ----------
    target : quarterly GDP growth, PeriodIndex(freq='Q'), NaNs dropped before use
    forecast_fn : callable with one of two signatures:
        - forecast_fn(y_train: np.ndarray) -> float
          (for quarterly-only models: AR, RW)
        - forecast_fn(y_train: np.ndarray, X_vintage: pd.DataFrame) -> float
          (for mixed-frequency models; X_vintage is the ragged-edge panel for that vintage)
    oos_start : first target quarter (default config.OOS_START)
    monthly_panel : full transformed monthly panel; required when forecast_fn uses X_vintage
    vintage : 1, 2, or 3; determines the as-of date for ragged-edge masking

    Returns
    -------
    DataFrame indexed by target quarter with columns: forecast, realized, error
    """
    oos_period = pd.Period(oos_start, freq="Q")
    target_clean = target.dropna()
    target_quarters = target_clean[target_clean.index >= oos_period].index

    use_panel = monthly_panel is not None and vintage is not None

    records = []
    for q in target_quarters:
        y_train = target_clean[target_clean.index < q].values
        if len(y_train) < 8:  # need enough history for BIC lag selection
            continue

        realized = float(target_clean[q])

        if use_panel:
            as_of = vintage_as_of(q, vintage)
            X_vintage = mask_ragged_edge(monthly_panel, as_of)
            forecast_val = forecast_fn(y_train, X_vintage)
        else:
            forecast_val = forecast_fn(y_train)

        records.append(
            {"period": q, "forecast": forecast_val, "realized": realized}
        )

    df = pd.DataFrame(records).set_index("period")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    df["error"] = df["realized"] - df["forecast"]
    return df


def compute_metrics(
    forecasts_df: pd.DataFrame,
    sample: str = "full",
) -> dict:
    """
    Compute RMSE, MAE, and mean bias.

    sample:
        'full'       — all rows
        'pre_covid'  — up to and including 2019Q4
        'ex_covid'   — exclude 2020Q1–2021Q2
    """
    df = forecasts_df.copy()

    if sample == "pre_covid":
        df = df[df.index <= pd.Period(COVID_START, freq="Q") - 1]
    elif sample == "ex_covid":
        covid_s = pd.Period(COVID_START, freq="Q")
        covid_e = pd.Period(COVID_END, freq="Q")
        df = df[(df.index < covid_s) | (df.index > covid_e)]

    e = df["error"].dropna()
    return {
        "sample": sample,
        "n_quarters": len(e),
        "rmse": float(np.sqrt((e**2).mean())),
        "mae": float(e.abs().mean()),
        "bias": float(e.mean()),
    }
