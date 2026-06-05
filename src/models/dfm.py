"""
Dynamic Factor Model nowcasts via statsmodels.tsa.statespace.DynamicFactorMQ.

The model jointly estimates one latent factor from all 12 monthly indicators
plus quarterly GDP growth.  The Kalman filter propagates the factor state
through months where indicators are NaN (ragged edge), so no imputation is
needed — missing values are handled natively.

Ragged-edge construction:
    For each (target quarter, vintage) pair the monthly panel is built to
    include ALL three months of the target quarter.  Unreleased cells are set
    to NaN; months that have not yet occurred (e.g. month 3 at vintage 1) are
    also all-NaN.  Quarterly GDP is NaN for the target quarter.  The Kalman
    filter projects the factor through the NaN months and the prediction at
    the quarter-end position is the nowcast.

Refit strategy (runtime trade-off):
    The EM algorithm re-estimates all state-space parameters every
    DFM_REFIT_EVERY=4 quarters.  Between refits, the Kalman smoother is run
    with the cached parameter vector on the updated dataset — one fast pass
    through the data with no optimisation.  This reduces runtime by ~75% with
    negligible nowcast accuracy cost.

Outputs:
    results/forecasts/dfm_v{1,2,3}.parquet
    results/tables/dfm_factor_loadings.csv
    results/tables/dfm_metrics.csv

Usage:
    python -m src.models.dfm
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from src.config import (
    DFM_N_FACTORS,
    DFM_REFIT_EVERY,
    FORECASTS_DIR,
    OOS_START,
    PREDICTOR_SERIES,
    TABLES_DIR,
)
from src.data.ragged_edge import mask_ragged_edge, vintage_as_of
from src.data.transforms import load_monthly_panel, load_quarterly_target
from src.evaluation.cv import compute_metrics

VINTAGES = [1, 2, 3]


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def _build_dfm_inputs(
    monthly_panel: pd.DataFrame,
    target: pd.Series,
    q: pd.Period,
    vintage: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the monthly and quarterly DataFrames for DynamicFactorMQ.

    Monthly panel:
        Includes all months up to (and including) the last month of quarter q.
        Cells unreleased on the vintage date are NaN; months that haven't
        occurred yet (e.g. month 3 at vintage 1) are all-NaN rows.
        The Kalman filter treats all NaN cells as missing observations and
        propagates the factor state through them using the AR dynamics.

    Quarterly GDP:
        Includes all completed quarters up to q-1 with realized values.
        Quarter q itself is NaN — this is the value being nowcast.
    """
    as_of = vintage_as_of(q, vintage)

    # Last month of the target quarter
    last_month_of_q = q.to_timestamp("Q").to_period("M")  # e.g. 2024Q1 -> 2024-03

    # Apply ragged-edge mask then extend to include all months through quarter-end
    masked = mask_ragged_edge(monthly_panel, as_of)
    monthly_input = masked.reindex(
        pd.period_range(masked.index[0], last_month_of_q, freq="M")
    )  # adds future months as all-NaN rows if they don't yet appear

    # Convert to DatetimeIndex (statsmodels requires this)
    monthly_df = monthly_input.copy()
    monthly_df.index = monthly_input.index.to_timestamp("M")
    monthly_df = monthly_df.asfreq("ME")  # month-end frequency

    # Quarterly GDP: set target quarter to NaN
    gdp_input = target.reindex(target.index[target.index <= q]).copy()
    gdp_input.at[q] = np.nan  # mask — this is what we're forecasting

    quarterly_df = gdp_input.to_frame(name="gdp_growth")
    quarterly_df.index = gdp_input.index.to_timestamp("Q")
    quarterly_df = quarterly_df.asfreq("QE")

    return monthly_df, quarterly_df


# ---------------------------------------------------------------------------
# Nowcast extraction
# ---------------------------------------------------------------------------

def _extract_nowcast(result, q: pd.Period) -> float:
    """
    Extract the DFM nowcast for quarter q.

    We use the Kalman smoother's predictions (Z * alpha_{t|T}) rather than
    the one-step-ahead filter forecasts (Z * alpha_{t|t-1}).  For the target
    quarter's last month, where GDP is NaN, the smoothed prediction
    incorporates all within-quarter monthly indicator updates — the one-step-
    ahead forecast does not, because it uses the state from the previous month.

    Since the dataset ends at the target quarter (no future data), smoother and
    filter agree on the final state; the difference is only in the measurement
    prediction at month-end when monthly indicators are also observed.
    """
    q_ts = q.to_timestamp("Q")

    # Primary: smoothed forecasts Z * alpha_{t|T} for the GDP variable.
    # Uses result.model.data.dates (not result.model.dates) and the
    # smoother_results.smoothed_forecasts array (n_endog, n_obs).
    try:
        sm = result.smoother_results
        sf = sm.smoothed_forecasts  # (n_endog, n_obs) — may raise if not computed
        if sf is not None:
            model_dates = pd.DatetimeIndex(result.model.data.dates)
            if q_ts in model_dates:
                t_idx = model_dates.get_loc(q_ts)
                endog_names = result.model.endog_names
                gdp_idx = next(
                    (i for i, n in enumerate(endog_names) if "gdp" in n.lower()), None
                )
                if gdp_idx is not None:
                    val = float(sf[gdp_idx, t_idx])
                    if np.isfinite(val):
                        return val
    except Exception:
        pass

    # Fallback: one-step-ahead from fittedvalues
    try:
        fv = result.fittedvalues
        if "gdp_growth" in fv.columns and q_ts in fv.index:
            val = float(fv.loc[q_ts, "gdp_growth"])
            if np.isfinite(val):
                return val
    except Exception:
        pass

    return np.nan


# ---------------------------------------------------------------------------
# Factor loadings export
# ---------------------------------------------------------------------------

def _save_factor_loadings(result) -> None:
    """Parse parameter names and save factor loadings to a CSV."""
    try:
        params = pd.Series(result.params, index=result.param_names)
        loadings = params[[n for n in params.index if "loading" in n]]
        df = loadings.rename_axis("param").reset_index()
        df.columns = ["param", "loading"]
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES_DIR / "dfm_factor_loadings.csv", index=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, name: str) -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.astype(str)
    out.to_parquet(FORECASTS_DIR / f"{name}.parquet")


def _load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(FORECASTS_DIR / f"{name}.parquet")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_dfm() -> pd.DataFrame:
    """
    Run DFM expanding-window OOS evaluation for all three vintages.
    Returns metrics DataFrame; saves forecast parquets, factor loadings, metrics CSV.
    """
    monthly_panel = load_monthly_panel()
    target = load_quarterly_target()

    oos_start = pd.Period(OOS_START, freq="Q")
    oos_quarters = target.dropna().index[target.dropna().index >= oos_start]

    metrics_rows: list[dict] = []
    loadings_saved = False

    for v in VINTAGES:
        print(f"  Vintage {v} ...")
        records: list[dict] = []

        cached_params: np.ndarray | None = None
        last_fit_idx: int = -DFM_REFIT_EVERY  # force refit on first quarter

        for idx, q in enumerate(oos_quarters):
            realized = float(target.at[q])
            monthly_df, quarterly_df = _build_dfm_inputs(monthly_panel, target, q, v)

            should_refit = (idx - last_fit_idx) >= DFM_REFIT_EVERY

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = DynamicFactorMQ(
                        monthly_df,
                        endog_quarterly=quarterly_df,
                        k_factors=DFM_N_FACTORS,
                        factor_orders=1,
                    )

                    if should_refit or cached_params is None:
                        result = model.fit(
                            disp=False,
                            maxiter=200,
                            em_initialization=True,
                        )
                        cached_params = result.params.copy()
                        last_fit_idx = idx

                        # Save factor loadings from the very first successful fit
                        if not loadings_saved:
                            _save_factor_loadings(result)
                            loadings_saved = True
                    else:
                        # Fast path: Kalman smoother with cached parameters, no EM
                        result = model.smooth(cached_params)

                fcst = _extract_nowcast(result, q)

            except Exception as exc:
                # Non-convergence or data issue — skip this quarter
                fcst = np.nan

            error = (realized - fcst) if np.isfinite(fcst) else np.nan
            records.append(
                {"period": q, "forecast": fcst, "realized": realized, "error": error}
            )

        df = pd.DataFrame(records).set_index("period")
        df.index = pd.PeriodIndex(df.index, freq="Q")
        _save(df, f"dfm_v{v}")

        for sample in ("full", "pre_covid", "ex_covid"):
            m = compute_metrics(df, sample=sample)
            m["model"] = "dfm"
            m["vintage"] = v
            metrics_rows.append(m)

        n_fcst = df["forecast"].notna().sum()
        print(f"    Quarters with valid nowcast: {n_fcst}/{len(df)}")

    metrics = pd.DataFrame(metrics_rows)[
        ["model", "vintage", "sample", "n_quarters", "rmse", "mae", "bias"]
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_DIR / "dfm_metrics.csv", index=False)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Dynamic Factor Model ...")
    print(f"  (Refitting parameters every {DFM_REFIT_EVERY} quarters; smoother otherwise)")
    metrics = run_dfm()

    print("\nDFM RMSE by vintage and sample:")
    print(metrics[["vintage", "sample", "n_quarters", "rmse", "mae"]].to_string(index=False))

    print("\nFactor loadings -> results/tables/dfm_factor_loadings.csv")
    print("Metrics        -> results/tables/dfm_metrics.csv")
