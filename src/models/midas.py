"""
U-MIDAS (Unrestricted MIDAS) regression nowcasts.

For each monthly predictor, the design matrix contains K=3 monthly observations
within the target quarter (the three months of the quarter being nowcast) plus
one lag of quarterly GDP growth.  OLS is fit with no polynomial restriction on
the monthly weights — hence "unrestricted".

Per-indicator models are combined by simple averaging into a combination forecast.

Ragged-edge handling:
    The masked monthly panel (NaN = not yet released) is forward-filled (LOCF)
    before extracting within-quarter features.  This carries the most recent
    available observation forward into unreleased slots so the design matrix
    always has K=3 values.  LOCF is applied only to columns that have a prior
    observation; the very first observation of a short-history series may still
    be NaN (handled by dropping that training row).

Almon polynomial MIDAS is not implemented (marked as future work); U-MIDAS is
the right starting point given K=3 and ~60+ training observations per window.

Outputs (per vintage v):
    results/forecasts/midas_{SERIES}_{v}.parquet    per-indicator forecasts
    results/forecasts/midas_combination_{v}.parquet  simple average combination
    results/tables/midas_metrics.csv

Usage:
    python -m src.models.midas
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FORECASTS_DIR, MIDAS_K, OOS_START, PREDICTOR_SERIES, TABLES_DIR
from src.data.ragged_edge import mask_ragged_edge, vintage_as_of
from src.data.transforms import load_monthly_panel, load_quarterly_target
from src.evaluation.cv import compute_metrics

VINTAGES = [1, 2, 3]
K = MIDAS_K  # 3 monthly lags per quarter


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _precompute_training_features(
    monthly_panel: pd.DataFrame,
    target: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    For each indicator, build a quarterly DataFrame with K monthly lags
    (m1=first month of quarter, m2=second, m3=third), the target y, and y_lag.

    Rows where any feature or label is NaN are kept as-is; dropna() is called
    at fit time so the training window can differ per indicator.
    """
    y_lag = target.shift(1)
    all_quarters = target.dropna().index
    features: dict[str, pd.DataFrame] = {}

    for sid in PREDICTOR_SERIES:
        series = monthly_panel[sid]
        rows = []
        for q in all_quarters:
            q_mask = series.index.to_timestamp().to_period("Q") == q
            x_vals = series[q_mask].values  # typically 3 values
            if len(x_vals) < K:
                continue
            rows.append(
                {
                    "period": q,
                    "m1": x_vals[0],
                    "m2": x_vals[1],
                    "m3": x_vals[2],
                    "y": float(target.get(q, np.nan)),
                    "y_lag": float(y_lag.get(q, np.nan)),
                }
            )
        df = pd.DataFrame(rows).set_index("period")
        df.index = pd.PeriodIndex(df.index, freq="Q")
        features[sid] = df

    return features


def _forecast_features(
    monthly_panel: pd.DataFrame,
    target_quarter: pd.Period,
    vintage: int,
    y_lag_q: float,
) -> dict[str, np.ndarray]:
    """
    Return per-indicator K-element feature arrays for the forecast quarter at
    the given vintage.  LOCF fills unreleased months with the last known value.
    """
    as_of = vintage_as_of(target_quarter, vintage)
    masked = mask_ragged_edge(monthly_panel, as_of)
    filled = masked.ffill()  # LOCF: carry last observed value forward

    q_mask = filled.index.to_timestamp().to_period("Q") == target_quarter
    q_rows = filled[q_mask]  # exactly 3 rows for any complete quarter in the panel

    result: dict[str, np.ndarray] = {}
    for sid in PREDICTOR_SERIES:
        vals = q_rows[sid].values[:K] if len(q_rows) >= K else np.full(K, np.nan)
        result[sid] = vals
    return result


# ---------------------------------------------------------------------------
# OLS fit and predict
# ---------------------------------------------------------------------------

def _umidas_ols(
    X_monthly: np.ndarray,  # (n, K) — monthly lags
    y_lag: np.ndarray,       # (n,)   — lagged quarterly GDP
    y: np.ndarray,           # (n,)   — target quarterly GDP
    x_fcst: np.ndarray,      # (K,)   — forecast-quarter monthly lags
    y_lag_fcst: float,
) -> float:
    """Fit U-MIDAS OLS and return one-step-ahead forecast."""
    X = np.column_stack([np.ones(len(y)), X_monthly, y_lag])
    if X.shape[0] < X.shape[1] + 1:  # under-determined system
        return np.nan
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    feat = np.concatenate([[1.0], x_fcst, [y_lag_fcst]])
    return float(feat @ beta)


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

def run_midas() -> pd.DataFrame:
    """
    Run U-MIDAS for all indicators and vintages.
    Returns metrics DataFrame; saves forecast parquets and midas_metrics.csv.
    """
    monthly_panel = load_monthly_panel()
    target = load_quarterly_target()
    y_lag_series = target.shift(1)

    print("  Precomputing training feature tables ...")
    train_feats = _precompute_training_features(monthly_panel, target)

    oos_start = pd.Period(OOS_START, freq="Q")
    oos_quarters = target.dropna().index[target.dropna().index >= oos_start]

    metrics_rows = []

    for v in VINTAGES:
        print(f"  Vintage {v} ...")
        indicator_records: dict[str, list] = {sid: [] for sid in PREDICTOR_SERIES}
        combo_records: list = []

        for q in oos_quarters:
            realized = float(target.at[q])
            y_lag_q = float(y_lag_series.get(q, np.nan))

            # Forecast-quarter LOCF features (vintage-specific)
            fcst_feats = _forecast_features(monthly_panel, q, v, y_lag_q)

            valid_fcsts: list[float] = []

            for sid in PREDICTOR_SERIES:
                x_fcst = fcst_feats[sid]
                train_df = train_feats[sid][train_feats[sid].index < q].dropna()

                if (
                    len(train_df) < K + 3  # minimum: K lags + intercept + y_lag + some slack
                    or np.isnan(y_lag_q)
                    or np.any(np.isnan(x_fcst))
                ):
                    fcst = np.nan
                else:
                    fcst = _umidas_ols(
                        train_df[["m1", "m2", "m3"]].values,
                        train_df["y_lag"].values,
                        train_df["y"].values,
                        x_fcst,
                        y_lag_q,
                    )

                error = (realized - fcst) if not np.isnan(fcst) else np.nan
                indicator_records[sid].append(
                    {"period": q, "forecast": fcst, "realized": realized, "error": error}
                )
                if not np.isnan(fcst):
                    valid_fcsts.append(fcst)

            # Combination: simple average of valid per-indicator forecasts
            combo = float(np.mean(valid_fcsts)) if valid_fcsts else np.nan
            combo_error = (realized - combo) if not np.isnan(combo) else np.nan
            combo_records.append(
                {"period": q, "forecast": combo, "realized": realized, "error": combo_error}
            )

        # Save per-indicator files
        for sid, records in indicator_records.items():
            df = pd.DataFrame(records).set_index("period")
            df.index = pd.PeriodIndex(df.index, freq="Q")
            _save(df, f"midas_{sid}_{v}")

        # Save combination file
        combo_df = pd.DataFrame(combo_records).set_index("period")
        combo_df.index = pd.PeriodIndex(combo_df.index, freq="Q")
        _save(combo_df, f"midas_combination_{v}")

        # Metrics
        for sample in ("full", "pre_covid", "ex_covid"):
            m = compute_metrics(combo_df, sample=sample)
            m["model"] = "midas_combination"
            m["vintage"] = v
            metrics_rows.append(m)

        for sid in PREDICTOR_SERIES:
            df_ind = _load(f"midas_{sid}_{v}")
            m = compute_metrics(df_ind, sample="full")
            m["model"] = f"midas_{sid}"
            m["vintage"] = v
            metrics_rows.append(m)

    metrics = pd.DataFrame(metrics_rows)[
        ["model", "vintage", "sample", "n_quarters", "rmse", "mae", "bias"]
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_DIR / "midas_metrics.csv", index=False)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running U-MIDAS ...")
    metrics = run_midas()

    print("\nCombination MIDAS RMSE by vintage and sample:")
    combo = metrics[metrics["model"] == "midas_combination"].copy()
    print(combo[["vintage", "sample", "n_quarters", "rmse", "mae"]].to_string(index=False))

    print("\nBest single-indicator by full-sample RMSE (per vintage):")
    ind_full = metrics[
        (metrics["sample"] == "full") & (metrics["model"] != "midas_combination")
    ].copy()
    for v in VINTAGES:
        sub = ind_full[ind_full["vintage"] == v]
        if sub.empty:
            continue
        best = sub.loc[sub["rmse"].idxmin()]
        print(f"  Vintage {v}: {best['model']}  RMSE={best['rmse']:.4f}%")

    # Head-to-head vs bridge combination (pre-COVID)
    try:
        bridge_m = pd.read_csv(TABLES_DIR / "bridge_metrics.csv")
        print("\nMIDAS vs Bridge combination (pre-COVID RMSE):")
        print(f"  {'Vintage':<10} {'MIDAS':>8} {'Bridge':>8}")
        for v in VINTAGES:
            midas_rmse = metrics[
                (metrics["model"] == "midas_combination")
                & (metrics["vintage"] == v)
                & (metrics["sample"] == "pre_covid")
            ]["rmse"].iloc[0]
            bridge_rmse = bridge_m[
                (bridge_m["model"] == "bridge_combination")
                & (bridge_m["vintage"] == v)
                & (bridge_m["sample"] == "pre_covid")
            ]["rmse"].iloc[0]
            print(f"  Vintage {v}    {midas_rmse:>8.4f}  {bridge_rmse:>8.4f}")
    except Exception:
        pass

    print("\nMetrics -> results/tables/midas_metrics.csv")
