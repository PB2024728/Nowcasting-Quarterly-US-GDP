"""
Regularized regression nowcasts: Lasso and ElasticNet.

The design matrix stacks K=3 monthly lags for all 12 predictors (36 features)
plus one lag of quarterly GDP growth = 37 features total.  This is a
multi-indicator U-MIDAS setup.

Both models use scikit-learn Pipelines:
    Pipeline([StandardScaler, LassoCV / ElasticNetCV])

The scaler is refit at each expanding-window step.  Hyperparameters (alpha for
Lasso; alpha + l1_ratio for ElasticNet) are chosen by LassoCV / ElasticNetCV
with TimeSeriesSplit cross-validation — no plain KFold, no shuffling.

Ragged-edge handling: same LOCF approach as MIDAS (masked panel forward-filled
before extracting within-quarter monthly values).

COVID handling: no dummies added here; the interaction of regularization with
COVID outliers is analyzed in Day 9 (with/without dummies comparison).

Outputs:
    results/forecasts/lasso_v{1,2,3}.parquet
    results/forecasts/elasticnet_v{1,2,3}.parquet
    results/forecasts/lasso_selection.csv   (selection stability log)
    results/tables/regularized_metrics.csv

Usage:
    python -m src.models.regularized
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FORECASTS_DIR,
    LASSO_CV_SPLITS,
    MIDAS_K,
    OOS_START,
    PREDICTOR_SERIES,
    TABLES_DIR,
)
from src.data.ragged_edge import mask_ragged_edge, vintage_as_of
from src.data.transforms import load_monthly_panel, load_quarterly_target
from src.evaluation.cv import compute_metrics

VINTAGES = [1, 2, 3]
K = MIDAS_K  # 3 monthly lags per indicator

# Hyperparameter search grids
_ALPHA_GRID = np.logspace(-4, 1, 25)
_L1_RATIO_GRID = [0.1, 0.5, 0.7, 0.9, 0.95]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_COVID_DUMMIES = ["covid_2020q2", "covid_2020q3"]
_COVID_QUARTERS = {
    "covid_2020q2": pd.Period("2020Q2", "Q"),
    "covid_2020q3": pd.Period("2020Q3", "Q"),
}


def _feature_names(k: int = K, with_covid_dummies: bool = False) -> list[str]:
    base = [f"{sid}_m{j}" for sid in PREDICTOR_SERIES for j in range(1, k + 1)] + ["gdp_lag"]
    return base + _COVID_DUMMIES if with_covid_dummies else base


def _precompute_feature_matrix(
    monthly_panel: pd.DataFrame,
    target: pd.Series,
    k: int = K,
    with_covid_dummies: bool = False,
) -> pd.DataFrame:
    """
    Build quarterly DataFrame with k monthly lags per indicator + gdp_lag + y.
    Optionally appends binary COVID quarter dummies (2020Q2, 2020Q3).
    Rows with missing months retain NaN and are dropped at fit time via dropna().
    """
    y_lag = target.shift(1)
    rows = []

    for q in target.dropna().index:
        row: dict = {"period": q}
        for sid in PREDICTOR_SERIES:
            series = monthly_panel[sid]
            q_mask = series.index.to_timestamp().to_period("Q") == q
            x_vals = series[q_mask].values
            for j in range(1, k + 1):
                row[f"{sid}_m{j}"] = x_vals[j - 1] if len(x_vals) >= j else np.nan
        row["gdp_lag"] = float(y_lag.get(q, np.nan))
        if with_covid_dummies:
            for name, cq in _COVID_QUARTERS.items():
                row[name] = 1.0 if q == cq else 0.0
        row["y"] = float(target.get(q, np.nan))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("period")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


def _forecast_feature_vector(
    monthly_panel: pd.DataFrame,
    q: pd.Period,
    vintage: int,
    y_lag_q: float,
    k: int = K,
    with_covid_dummies: bool = False,
) -> np.ndarray:
    """
    Build the feature vector for quarter q at a given vintage.
    Unreleased monthly cells are LOCF-filled from the most recent available value.
    """
    as_of = vintage_as_of(q, vintage)
    masked = mask_ragged_edge(monthly_panel, as_of)
    filled = masked.ffill()

    q_mask = filled.index.to_timestamp().to_period("Q") == q
    q_rows = filled[q_mask]

    feat: list[float] = []
    for sid in PREDICTOR_SERIES:
        vals = q_rows[sid].values if len(q_rows) >= k else np.full(k, np.nan)
        feat.extend(vals[:k].tolist())
    feat.append(y_lag_q)
    if with_covid_dummies:
        for name, cq in _COVID_QUARTERS.items():
            feat.append(1.0 if q == cq else 0.0)

    return np.array(feat, dtype=float)


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _make_lasso_pipe(tscv: TimeSeriesSplit) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LassoCV(
                    alphas=_ALPHA_GRID,
                    cv=tscv,
                    max_iter=10000,
                    fit_intercept=True,
                ),
            ),
        ]
    )


def _make_enet_pipe(tscv: TimeSeriesSplit) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    alphas=_ALPHA_GRID,
                    l1_ratio=_L1_RATIO_GRID,
                    cv=tscv,
                    max_iter=10000,
                    fit_intercept=True,
                ),
            ),
        ]
    )


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

def run_regularized(with_covid_dummies: bool = False) -> pd.DataFrame:
    """
    Run Lasso and ElasticNet expanding-window OOS for all three vintages.
    Returns metrics DataFrame; saves forecast parquets, selection log, and metrics CSV.
    """
    monthly_panel = load_monthly_panel()
    target = load_quarterly_target()
    y_lag_series = target.shift(1)

    suffix = "_covid" if with_covid_dummies else ""
    print("  Precomputing full feature matrix ...")
    feat_matrix = _precompute_feature_matrix(monthly_panel, target, with_covid_dummies=with_covid_dummies)
    feat_names = _feature_names(with_covid_dummies=with_covid_dummies)

    oos_start = pd.Period(OOS_START, freq="Q")
    oos_quarters = target.dropna().index[target.dropna().index >= oos_start]
    tscv = TimeSeriesSplit(n_splits=LASSO_CV_SPLITS)

    metrics_rows: list[dict] = []
    selection_rows: list[dict] = []

    for v in VINTAGES:
        print(f"  Vintage {v} ...")
        lasso_records: list[dict] = []
        enet_records: list[dict] = []

        for q in oos_quarters:
            realized = float(target.at[q])
            y_lag_q = float(y_lag_series.get(q, np.nan))

            # Training data
            train_df = feat_matrix[feat_matrix.index < q].dropna()

            # Forecast feature vector (LOCF)
            x_fcst = _forecast_feature_vector(monthly_panel, q, v, y_lag_q, with_covid_dummies=with_covid_dummies)
            has_nan = np.isnan(y_lag_q) or np.any(np.isnan(x_fcst)) or len(train_df) < LASSO_CV_SPLITS * 4

            X_train = train_df[feat_names].values
            y_train = train_df["y"].values

            # ----- Lasso -----
            if has_nan:
                lasso_fcst = np.nan
                sel_row = {"quarter": str(q), "vintage": v, "best_alpha": np.nan, "n_selected": np.nan}
                sel_row.update({f: np.nan for f in feat_names})
            else:
                pipe_lasso = _make_lasso_pipe(tscv)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe_lasso.fit(X_train, y_train)
                lasso_fcst = float(pipe_lasso.predict(x_fcst.reshape(1, -1))[0])

                coefs = pipe_lasso.named_steps["model"].coef_
                alpha_best = float(pipe_lasso.named_steps["model"].alpha_)
                selected = (coefs != 0.0).astype(int)
                sel_row = {
                    "quarter": str(q),
                    "vintage": v,
                    "best_alpha": alpha_best,
                    "n_selected": int(selected.sum()),
                }
                sel_row.update(dict(zip(feat_names, selected.tolist())))

            selection_rows.append(sel_row)
            lasso_error = (realized - lasso_fcst) if not np.isnan(lasso_fcst) else np.nan
            lasso_records.append(
                {"period": q, "forecast": lasso_fcst, "realized": realized, "error": lasso_error}
            )

            # ----- ElasticNet -----
            if has_nan:
                enet_fcst = np.nan
            else:
                pipe_enet = _make_enet_pipe(tscv)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe_enet.fit(X_train, y_train)
                enet_fcst = float(pipe_enet.predict(x_fcst.reshape(1, -1))[0])

            enet_error = (realized - enet_fcst) if not np.isnan(enet_fcst) else np.nan
            enet_records.append(
                {"period": q, "forecast": enet_fcst, "realized": realized, "error": enet_error}
            )

        # Save forecast files
        for model_name, records in [("lasso", lasso_records), ("elasticnet", enet_records)]:
            df = pd.DataFrame(records).set_index("period")
            df.index = pd.PeriodIndex(df.index, freq="Q")
            _save(df, f"{model_name}_v{v}{suffix}")

            for sample in ("full", "pre_covid", "ex_covid"):
                m = compute_metrics(df, sample=sample)
                m["model"] = f"{model_name}{suffix}"
                m["vintage"] = v
                metrics_rows.append(m)

    # Save Lasso variable-selection log (base run only)
    if not with_covid_dummies:
        sel_df = pd.DataFrame(selection_rows)
        FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        sel_df.to_csv(FORECASTS_DIR / "lasso_selection.csv", index=False)

    metrics = pd.DataFrame(metrics_rows)[
        ["model", "vintage", "sample", "n_quarters", "rmse", "mae", "bias"]
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fname = "regularized_metrics_covid.csv" if with_covid_dummies else "regularized_metrics.csv"
    metrics.to_csv(TABLES_DIR / fname, index=False)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running regularized regression (Lasso + ElasticNet) ...")
    metrics = run_regularized()

    print("\nLasso RMSE by vintage and sample:")
    lasso_m = metrics[metrics["model"] == "lasso"].copy()
    print(lasso_m[["vintage", "sample", "n_quarters", "rmse", "mae"]].to_string(index=False))

    print("\nElasticNet RMSE by vintage and sample:")
    enet_m = metrics[metrics["model"] == "elasticnet"].copy()
    print(enet_m[["vintage", "sample", "n_quarters", "rmse", "mae"]].to_string(index=False))

    # Selection stability summary
    sel = pd.read_csv(FORECASTS_DIR / "lasso_selection.csv")
    print("\nLasso selection stability (% of quarters selected, averaged across vintages):")
    feat_names = _feature_names()
    sel_cols = [c for c in feat_names if c in sel.columns]
    stability = sel[sel_cols].mean().sort_values(ascending=False).head(10)
    for feat, pct in stability.items():
        print(f"  {feat:<22}  {pct * 100:.1f}%")

    print("\nOutputs -> results/forecasts/ and results/tables/regularized_metrics.csv")
