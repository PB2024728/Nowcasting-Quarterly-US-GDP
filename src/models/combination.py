"""
Method-family combination forecast.

Averages the best-combination forecast from each of the four model families:
    Bridge combination, MIDAS combination, ElasticNet, DFM

This implements the classic "forecast combination" approach: pooling forecasts
from diverse methods often reduces variance without increasing bias, producing
an ensemble that is more robust than any single model.

All three within-quarter vintages are combined separately.

Outputs:
    results/forecasts/combination_v{1,2,3}.parquet
    results/tables/combination_metrics.csv

Usage:
    python -m src.models.combination
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FORECASTS_DIR, OOS_START, TABLES_DIR
from src.evaluation.cv import compute_metrics
from src.evaluation.tests import dm_test

VINTAGES = [1, 2, 3]

# The four family representatives combined at each vintage
_FAMILY_FILES = {
    1: ["bridge_combination_1", "midas_combination_1", "elasticnet_v1", "dfm_v1"],
    2: ["bridge_combination_2", "midas_combination_2", "elasticnet_v2", "dfm_v2"],
    3: ["bridge_combination_3", "midas_combination_3", "elasticnet_v3", "dfm_v3"],
}


def _load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(FORECASTS_DIR / f"{name}.parquet")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


def _save(df: pd.DataFrame, name: str) -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.astype(str)
    out.to_parquet(FORECASTS_DIR / f"{name}.parquet")


def run_combination() -> pd.DataFrame:
    """
    Build and evaluate the method-family combination for all three vintages.
    Returns metrics DataFrame; saves forecast parquets and combination_metrics.csv.
    """
    oos_start = pd.Period(OOS_START, freq="Q")
    metrics_rows: list[dict] = []

    for v in VINTAGES:
        # Load each family's forecast
        family_dfs = [_load(fname) for fname in _FAMILY_FILES[v]]

        # Use bridge index as the master reference
        reference = family_dfs[0]
        common_idx = reference.index[reference.index >= oos_start]

        # Stack forecasts into a single DataFrame
        fc_matrix = pd.DataFrame(
            {fname: df.loc[common_idx, "forecast"] for fname, df in
             zip(_FAMILY_FILES[v], family_dfs)},
            index=common_idx,
        )
        realized = reference.loc[common_idx, "realized"]

        # Simple average (skip NaN — so if DFM is NaN for a quarter, use the other three)
        combo_forecast = fc_matrix.mean(axis=1, skipna=True)
        combo_error = realized - combo_forecast

        combo_df = pd.DataFrame(
            {"forecast": combo_forecast, "realized": realized, "error": combo_error}
        )
        combo_df.index.name = "period"
        _save(combo_df, f"combination_v{v}")

        # Metrics for all three samples
        for sample in ("full", "pre_covid", "ex_covid"):
            m = compute_metrics(combo_df, sample=sample)
            m["model"] = "combination"
            m["vintage"] = v
            metrics_rows.append(m)

    metrics = pd.DataFrame(metrics_rows)[
        ["model", "vintage", "sample", "n_quarters", "rmse", "mae", "bias"]
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_DIR / "combination_metrics.csv", index=False)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building method-family combination forecasts ...")
    metrics = run_combination()

    print("\nCombination RMSE by vintage and sample:")
    print(metrics.to_string(index=False))

    print("\nDM tests: combination vs. best individual (bridge_combination):")
    print(f"  {'Vintage':<10} {'Sample':<12} {'Comb RMSE':>10} {'Bridge RMSE':>12} {'DM stat':>9} {'p-value':>9}")
    print("  " + "-" * 65)
    for v in VINTAGES:
        combo = _load(f"combination_v{v}")
        bridge = _load(f"bridge_combination_{v}")
        common = combo.index.intersection(bridge.index)

        from src.evaluation.cv import compute_metrics as _cm
        from src.config import COVID_START, COVID_END

        for sample, label in [("pre_covid", "pre-COVID"), ("ex_covid", "ex-COVID")]:
            covid_s = pd.Period(COVID_START, "Q")
            covid_e = pd.Period(COVID_END, "Q")
            if sample == "pre_covid":
                mask = common < covid_s
            else:
                mask = (common < covid_s) | (common > covid_e)
            idx = common[mask]

            e_bridge = bridge.loc[idx, "error"].values
            e_combo = combo.loc[idx, "error"].values
            stat, pval, _ = dm_test(e_bridge, e_combo)

            b_rmse = float(np.sqrt((e_bridge**2).mean()))
            c_rmse = float(np.sqrt((e_combo**2).mean()))
            sig = "*" if pval < 0.10 else " "
            print(f"  V{v} {label:<11} {c_rmse:>10.4f} {b_rmse:>12.4f} {stat:>9.3f} {pval:>8.3f}{sig}")

    print("\nOutputs -> results/forecasts/combination_v{{1,2,3}}.parquet")
    print("         -> results/tables/combination_metrics.csv")
