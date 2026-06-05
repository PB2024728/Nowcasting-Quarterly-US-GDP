"""
Master evaluation script.

Loads every forecast parquet from results/forecasts/, computes RMSE, MAE,
and bias for each model-vintage combination, then runs Diebold-Mariano tests
of every alternative model against the AR(p) baseline.

Three sample windows are evaluated:
    full       — all OOS quarters (2005Q1 to present)
    pre_covid  — 2005Q1 through 2019Q4
    ex_covid   — full OOS excluding 2020Q1–2021Q2

COVID handling note:
    Regularized regression models are included twice: once without COVID
    dummies (lasso, elasticnet) and once with dummies added for 2020Q2 and
    2020Q3 (lasso_covid, elasticnet_covid).  The comparison shows whether
    explicitly flagging those outlier quarters improves out-of-sample accuracy
    once they enter the training set.  Pre-COVID results are identical by
    construction; the difference appears in full-sample and ex-COVID metrics.

Outputs:
    results/tables/master_results.csv   — full model × vintage × sample table
    results/tables/master_summary.csv   — key combination/aggregate models only

Usage:
    python -m src.evaluation.master
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FORECASTS_DIR, TABLES_DIR
from src.evaluation.cv import compute_metrics
from src.evaluation.tests import dm_test

# Models included in the concise summary table (combinations / aggregates)
_SUMMARY_MODELS = {
    "AR(p)", "RW",
    "bridge_combination",
    "midas_combination",
    "lasso", "lasso_covid",
    "elasticnet", "elasticnet_covid",
    "dfm",
    "combination",
}


# ---------------------------------------------------------------------------
# Filename → (model_name, vintage)
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> tuple[str, int | None]:
    """
    Map a forecast parquet filename stem to (model_label, vintage).
    Returns (None, None) for files that should be skipped (e.g. non-forecast).
    """
    if stem == "benchmark_ar":
        return "AR(p)", None
    if stem == "benchmark_rw":
        return "RW", None
    if stem == "lasso_selection":
        return None, None  # CSV, not a forecast file

    # dfm_v{1,2,3}
    m = re.fullmatch(r"dfm_v(\d)", stem)
    if m:
        return "dfm", int(m.group(1))

    # lasso_v{1,2,3}[_covid]
    m = re.fullmatch(r"(lasso|elasticnet)_v(\d)(_covid)?", stem)
    if m:
        label = m.group(1) + (m.group(3) or "")
        return label, int(m.group(2))

    # bridge_{SERIES}_{v} or bridge_combination_{v}
    m = re.fullmatch(r"(bridge_.+)_(\d)", stem)
    if m:
        return m.group(1), int(m.group(2))

    # midas_{SERIES}_{v} or midas_combination_{v}
    m = re.fullmatch(r"(midas_.+)_(\d)", stem)
    if m:
        return m.group(1), int(m.group(2))

    # combination_v{1,2,3}
    m = re.fullmatch(r"combination_v(\d)", stem)
    if m:
        return "combination", int(m.group(1))

    return stem, None  # fallback


# ---------------------------------------------------------------------------
# Load all forecasts
# ---------------------------------------------------------------------------

def _load_all_forecasts() -> dict[tuple[str, int | None], pd.DataFrame]:
    """Return {(model, vintage): DataFrame} for every parquet in FORECASTS_DIR."""
    result: dict[tuple[str, int | None], pd.DataFrame] = {}
    for path in sorted(FORECASTS_DIR.glob("*.parquet")):
        model, vintage = _parse_stem(path.stem)
        if model is None:
            continue
        df = pd.read_parquet(path)
        df.index = pd.PeriodIndex(df.index, freq="Q")
        result[(model, vintage)] = df
    return result


# ---------------------------------------------------------------------------
# Master evaluation
# ---------------------------------------------------------------------------

def run_master() -> pd.DataFrame:
    """
    Compute metrics and DM tests for all forecast files.
    Returns full results DataFrame; saves master_results.csv and master_summary.csv.
    """
    forecasts = _load_all_forecasts()
    print(f"  Loaded {len(forecasts)} forecast series.")

    # AR baseline (no vintage) — used for DM tests
    ar_df = forecasts.get(("AR(p)", None))
    if ar_df is None:
        raise FileNotFoundError("AR benchmark forecast not found. Run src.models.benchmarks first.")

    rows: list[dict] = []

    for (model, vintage), df in forecasts.items():
        for sample in ("full", "pre_covid", "ex_covid"):
            m = compute_metrics(df, sample=sample)
            m["model"] = model
            m["vintage"] = vintage if vintage is not None else "all"

            # Diebold-Mariano test vs AR(p)
            if model == "AR(p)":
                m["dm_stat"] = np.nan
                m["dm_pval"] = np.nan
                m["dm_mean_diff"] = np.nan
            else:
                # Align errors on common index
                common = df.index.intersection(ar_df.index)
                e_ar = ar_df.loc[common, "error"].values
                e_alt = df.loc[common, "error"].values

                # Restrict to sample window
                sample_mask = _sample_mask(ar_df.loc[common], sample)
                e_ar_s = e_ar[sample_mask]
                e_alt_s = e_alt[sample_mask]

                stat, pval, diff = dm_test(e_ar_s, e_alt_s, loss="squared")
                m["dm_stat"] = stat
                m["dm_pval"] = pval
                m["dm_mean_diff"] = diff  # positive → alternative beats AR

            rows.append(m)

    results = pd.DataFrame(rows)[
        ["model", "vintage", "sample", "n_quarters",
         "rmse", "mae", "bias", "dm_stat", "dm_pval", "dm_mean_diff"]
    ]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES_DIR / "master_results.csv", index=False)

    # Summary: key models only
    summary = results[results["model"].isin(_SUMMARY_MODELS)].copy()
    summary.to_csv(TABLES_DIR / "master_summary.csv", index=False)

    return results


def _sample_mask(df: pd.DataFrame, sample: str) -> np.ndarray:
    """Boolean mask for a sample window applied to a quarterly-indexed DataFrame."""
    from src.config import COVID_END, COVID_START
    idx = df.index
    if sample == "full":
        return np.ones(len(idx), dtype=bool)
    elif sample == "pre_covid":
        cutoff = pd.Period(COVID_START, "Q") - 1
        return np.array([p <= cutoff for p in idx], dtype=bool)
    elif sample == "ex_covid":
        covid_s = pd.Period(COVID_START, "Q")
        covid_e = pd.Period(COVID_END, "Q")
        return np.array([(p < covid_s) or (p > covid_e) for p in idx], dtype=bool)
    return np.ones(len(idx), dtype=bool)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running master evaluation ...")
    results = run_master()

    # Print summary for key combination models, pre-COVID sample
    print("\n=== Pre-COVID RMSE — key models ===")
    pre = results[
        (results["sample"] == "pre_covid") &
        (results["model"].isin(_SUMMARY_MODELS))
    ][["model", "vintage", "rmse", "dm_stat", "dm_pval"]].sort_values(
        ["vintage", "rmse"]
    )
    print(pre.to_string(index=False, float_format="{:.4f}".format))

    print("\n=== Significance vs AR(p) at 10% (pre-COVID) ===")
    sig = results[
        (results["sample"] == "pre_covid") &
        (results["dm_pval"] < 0.10) &
        (results["dm_stat"] > 0) &
        (results["model"].isin(_SUMMARY_MODELS))
    ][["model", "vintage", "rmse", "dm_stat", "dm_pval"]]
    if sig.empty:
        print("  None significant at 10%.")
    else:
        print(sig.to_string(index=False, float_format="{:.4f}".format))

    print(f"\nFull results -> results/tables/master_results.csv")
    print(f"Summary      -> results/tables/master_summary.csv")
