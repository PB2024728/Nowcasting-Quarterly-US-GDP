"""
Bridge equation nowcasts.

For each monthly predictor, aggregate to quarterly and fit OLS:
    GDP_growth_t = b0 + b1 * indicator_t + b2 * GDP_growth_{t-1}

Three within-quarter vintages are produced using the ragged-edge masker:
  vintage 1 = as-of end of month 1 of the target quarter
  vintage 2 = as-of end of month 2
  vintage 3 = as-of end of month 3 (full quarter known)

Partial aggregation: when fewer than 3 months of a quarter are available
at a given vintage, the quarterly "indicator" is the mean of the available
months. This is the standard bridge-equation simplification — the model is
trained on complete quarterly aggregates and applied to partial ones.

Outputs (per vintage v):
  results/forecasts/bridge_{SERIES}_{v}.parquet   — per-indicator forecasts
  results/forecasts/bridge_combination_{v}.parquet — simple average combination
  results/tables/bridge_metrics.csv

Usage:
    python -m src.models.bridge
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FORECASTS_DIR, OOS_START, PREDICTOR_SERIES, TABLES_DIR
from src.data.ragged_edge import mask_ragged_edge, vintage_as_of
from src.data.transforms import load_monthly_panel, load_quarterly_target
from src.evaluation.cv import compute_metrics

VINTAGES = [1, 2, 3]


# ---------------------------------------------------------------------------
# Monthly → quarterly aggregation
# ---------------------------------------------------------------------------

def aggregate_to_quarterly(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Average monthly observations to quarterly frequency.

    Each monthly period is mapped to its enclosing calendar quarter; then the
    three monthly values are averaged (skipna=True so partial quarters from
    short-history series like RSAFS still contribute what they have).

    Returns DataFrame with PeriodIndex(freq='Q').
    """
    qkeys = monthly_panel.index.to_timestamp().to_period("Q").astype(str)
    agg = monthly_panel.groupby(qkeys).mean()
    agg.index = pd.PeriodIndex(agg.index, freq="Q")
    agg.index.name = "period"
    return agg


# ---------------------------------------------------------------------------
# Single-indicator OLS bridge
# ---------------------------------------------------------------------------

def _ols_forecast(
    y_train: np.ndarray,
    x_train: np.ndarray,
    y_lag_train: np.ndarray,
    x_current: float,
    y_lag_current: float,
) -> float:
    """
    Fit OLS: y = b0 + b1*x + b2*y_lag, return one-step-ahead forecast.
    Returns NaN if fewer than 5 complete observations are available.
    """
    X = np.column_stack([np.ones(len(y_train)), x_train, y_lag_train])
    if X.shape[0] < 5:
        return np.nan
    beta, _, _, _ = np.linalg.lstsq(X, y_train, rcond=None)
    return float(np.dot([1.0, x_current, y_lag_current], beta))


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

def run_bridge() -> pd.DataFrame:
    """
    Run bridge equations for all indicators and vintages.
    Returns metrics DataFrame; also saves forecast parquets and bridge_metrics.csv.
    """
    monthly_panel = load_monthly_panel()
    target = load_quarterly_target()
    y_lag = target.shift(1)  # GDP growth one quarter back

    # Full historical quarterly aggregates (no masking) — used for training
    q_panel = aggregate_to_quarterly(monthly_panel)

    oos_start = pd.Period(OOS_START, freq="Q")
    oos_quarters = target.dropna().index[target.dropna().index >= oos_start]

    metrics_rows = []

    for v in VINTAGES:
        print(f"  Vintage {v} ...")
        indicator_records: dict[str, list] = {sid: [] for sid in PREDICTOR_SERIES}
        combo_records: list = []

        for q in oos_quarters:
            realized = float(target.at[q])

            # Partial monthly means for the current quarter at this vintage
            as_of = vintage_as_of(q, v)
            masked = mask_ragged_edge(monthly_panel, as_of)
            q_months_mask = masked.index.to_timestamp().to_period("Q") == q
            q_months = masked[q_months_mask]

            if len(q_months) == 0:
                x_current = pd.Series(np.nan, index=list(PREDICTOR_SERIES.keys()))
            else:
                x_current = q_months.mean()  # skipna=True → partial mean

            # y_{q-1}: last quarter's realized GDP growth (always complete)
            y_lag_q = float(y_lag.get(q, np.nan))

            # Training window: all quarters strictly before q
            past = q_panel.index[q_panel.index < q]
            y_t_all = target.reindex(past)
            y_lag_all = y_lag.reindex(past)

            valid_fcsts: list[float] = []

            for sid in PREDICTOR_SERIES:
                x_q_val = float(x_current.get(sid, np.nan))

                if np.isnan(x_q_val) or np.isnan(y_lag_q):
                    fcst = np.nan
                else:
                    x_t_all = q_panel[sid].reindex(past)
                    ok = ~(y_t_all.isna() | y_lag_all.isna() | x_t_all.isna())
                    if ok.sum() < 5:
                        fcst = np.nan
                    else:
                        fcst = _ols_forecast(
                            y_t_all[ok].values,
                            x_t_all[ok].values,
                            y_lag_all[ok].values,
                            x_q_val,
                            y_lag_q,
                        )

                error = (realized - fcst) if not np.isnan(fcst) else np.nan
                indicator_records[sid].append(
                    {"period": q, "forecast": fcst, "realized": realized, "error": error}
                )
                if not np.isnan(fcst):
                    valid_fcsts.append(fcst)

            # Combination: simple average of all valid single-indicator forecasts
            combo = float(np.mean(valid_fcsts)) if valid_fcsts else np.nan
            combo_error = (realized - combo) if not np.isnan(combo) else np.nan
            combo_records.append(
                {"period": q, "forecast": combo, "realized": realized, "error": combo_error}
            )

        # Save per-indicator forecast files
        for sid, records in indicator_records.items():
            df = pd.DataFrame(records).set_index("period")
            df.index = pd.PeriodIndex(df.index, freq="Q")
            _save(df, f"bridge_{sid}_{v}")

        # Save combination forecast
        combo_df = pd.DataFrame(combo_records).set_index("period")
        combo_df.index = pd.PeriodIndex(combo_df.index, freq="Q")
        _save(combo_df, f"bridge_combination_{v}")

        # Metrics for combination
        for sample in ("full", "pre_covid", "ex_covid"):
            m = compute_metrics(combo_df, sample=sample)
            m["model"] = f"bridge_combination"
            m["vintage"] = v
            metrics_rows.append(m)

        # Metrics for each indicator (full sample only — used to find best)
        for sid in PREDICTOR_SERIES:
            df = _load(f"bridge_{sid}_{v}")
            m = compute_metrics(df, sample="full")
            m["model"] = f"bridge_{sid}"
            m["vintage"] = v
            metrics_rows.append(m)

    metrics = pd.DataFrame(metrics_rows)[
        ["model", "vintage", "sample", "n_quarters", "rmse", "mae", "bias"]
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_DIR / "bridge_metrics.csv", index=False)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running bridge equations ...")
    metrics = run_bridge()

    print("\nCombination bridge RMSE by vintage and sample:")
    combo = metrics[metrics["model"] == "bridge_combination"].copy()
    print(combo[["vintage", "sample", "n_quarters", "rmse", "mae"]].to_string(index=False))

    print("\nBest single-indicator by full-sample RMSE (per vintage):")
    indicator_full = metrics[
        (metrics["sample"] == "full") & (metrics["model"] != "bridge_combination")
    ].copy()
    for v in VINTAGES:
        sub = indicator_full[indicator_full["vintage"] == v]
        if sub.empty:
            continue
        best = sub.loc[sub["rmse"].idxmin()]
        print(f"  Vintage {v}: {best['model']}  RMSE={best['rmse']:.4f}%")

    print("\nMetrics -> results/tables/bridge_metrics.csv")
