"""
AR(p) and random-walk benchmark nowcasts.

AR(p): lag order p selected by BIC at each expanding window step.
RW:    forecast = historical mean of quarterly GDP growth up to the cutoff.

Usage:
    python -m src.models.benchmarks
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from src.config import AR_MAX_LAGS, FORECASTS_DIR, TABLES_DIR
from src.data.transforms import load_quarterly_target
from src.evaluation.cv import compute_metrics, expanding_window_oos


# ---------------------------------------------------------------------------
# Forecast functions
# ---------------------------------------------------------------------------

def _select_ar_lag(y: np.ndarray, max_lags: int) -> int:
    """BIC-optimal AR lag order, capped so p < len(y)/4."""
    max_p = max(1, min(max_lags, len(y) // 4 - 1))
    best_bic = np.inf
    best_p = 1
    for p in range(1, max_p + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = AutoReg(y, lags=p, old_names=False).fit()
            if res.bic < best_bic:
                best_bic = res.bic
                best_p = p
        except Exception:
            pass
    return best_p


def ar_forecast(y_train: np.ndarray) -> float:
    """One-step-ahead AR(p) forecast; p chosen by BIC on each training window."""
    p = _select_ar_lag(y_train, AR_MAX_LAGS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = AutoReg(y_train, lags=p, old_names=False).fit()
    pred = res.predict(start=len(y_train), end=len(y_train))
    return float(np.asarray(pred).flat[0])


def rw_forecast(y_train: np.ndarray) -> float:
    """Random-walk benchmark: historical mean of training data."""
    return float(np.mean(y_train))


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_forecasts(df: pd.DataFrame, name: str) -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.astype(str)
    out.to_parquet(FORECASTS_DIR / f"{name}.parquet")


def load_forecasts(name: str) -> pd.DataFrame:
    df = pd.read_parquet(FORECASTS_DIR / f"{name}.parquet")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmarks() -> pd.DataFrame:
    """
    Run AR and RW expanding-window OOS loops, save forecast parquets,
    compute metrics, and return the metrics DataFrame.
    """
    target = load_quarterly_target()

    print("Running AR(p) benchmark ...")
    ar_fcsts = expanding_window_oos(target, ar_forecast)
    _save_forecasts(ar_fcsts, "benchmark_ar")
    print(f"  Quarters evaluated: {len(ar_fcsts)}")

    print("Running random-walk benchmark ...")
    rw_fcsts = expanding_window_oos(target, rw_forecast)
    _save_forecasts(rw_fcsts, "benchmark_rw")

    # Compute metrics for both models across two samples
    rows = []
    for label, df in [("AR(p)", ar_fcsts), ("RW-mean", rw_fcsts)]:
        for sample in ("full", "pre_covid", "ex_covid"):
            m = compute_metrics(df, sample=sample)
            m["model"] = label
            rows.append(m)

    metrics = pd.DataFrame(rows)[
        ["model", "sample", "n_quarters", "rmse", "mae", "bias"]
    ]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_DIR / "benchmarks.csv", index=False)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metrics = run_benchmarks()
    print("\nBenchmark metrics:")
    print(metrics.to_string(index=False, float_format="{:.4f}".format))
    print("\nForecasts  -> results/forecasts/benchmark_{ar,rw}.parquet")
    print("Metrics    -> results/tables/benchmarks.csv")
    ar_full = metrics.loc[
        (metrics["model"] == "AR(p)") & (metrics["sample"] == "full"), "rmse"
    ].iloc[0]
    print(f"\nAR(p) full-sample RMSE (the bar to beat): {ar_full:.4f}%")
