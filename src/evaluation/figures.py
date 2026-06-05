"""
Publication-quality figures for the GDP nowcasting project.

Figure 1 — RMSE by model and vintage (grouped bar chart)
Figure 2 — Realized GDP growth vs. key model nowcasts (time series)
Figure 3 — RMSE as a function of within-quarter data arrival (line chart)
Figure 4 — Lasso variable selection stability (horizontal bar chart)

All figures saved to results/figures/ as PNG (300 DPI) and PDF.

Usage:
    python -m src.evaluation.figures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from src.config import COVID_END, COVID_START, FIGURES_DIR, FORECASTS_DIR, TABLES_DIR
from src.utils.plotting import (
    COLORS, MODEL_LABELS, VINTAGE_COLORS, save_fig, set_style,
)

set_style()

_KEY_MODELS = [
    "bridge_combination",
    "midas_combination",
    "lasso",
    "elasticnet",
    "dfm",
]

_VINTAGE_LABELS = {1: "Month 1", 2: "Month 2", 3: "Month 3"}


# ---------------------------------------------------------------------------
# Helper: load master summary
# ---------------------------------------------------------------------------

def _load_summary() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "master_summary.csv")
    # vintage is int for vintage models, 'all' for benchmarks
    df["vintage"] = pd.to_numeric(df["vintage"], errors="coerce")
    return df


def _load_forecast(name: str) -> pd.DataFrame:
    df = pd.read_parquet(FORECASTS_DIR / f"{name}.parquet")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


# ---------------------------------------------------------------------------
# Figure 1: RMSE by model and vintage
# ---------------------------------------------------------------------------

def fig1_rmse_by_model_vintage(sample: str = "pre_covid") -> plt.Figure:
    """
    Grouped bar chart: one group per model, three bars per group (vintages).
    AR(p) and RW shown as horizontal reference lines.
    """
    summary = _load_summary()
    pre = summary[summary["sample"] == sample]

    # Reference lines
    ar_rmse = float(pre.loc[pre["model"] == "AR(p)", "rmse"].iloc[0])
    rw_rmse = float(pre.loc[pre["model"] == "RW", "rmse"].iloc[0])

    # Model data
    model_data = pre[pre["model"].isin(_KEY_MODELS) & pre["vintage"].notna()]

    n_models = len(_KEY_MODELS)
    n_vintages = 3
    bar_w = 0.22
    group_gap = 0.9
    x = np.arange(n_models) * group_gap

    fig, ax = plt.subplots(figsize=(11, 5))

    for vi, (v, label) in enumerate(_VINTAGE_LABELS.items()):
        offsets = x + (vi - 1) * bar_w
        heights = [
            float(
                model_data.loc[
                    (model_data["model"] == m) & (model_data["vintage"] == v), "rmse"
                ].values[0]
            )
            if len(
                model_data.loc[
                    (model_data["model"] == m) & (model_data["vintage"] == v), "rmse"
                ]
            ) > 0
            else np.nan
            for m in _KEY_MODELS
        ]
        bars = ax.bar(
            offsets, heights, width=bar_w, color=VINTAGE_COLORS[vi],
            label=label, zorder=3,
        )
        # Value labels on bars
        for bar, h in zip(bars, heights):
            if np.isfinite(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7, color="#333333",
                )

    # Reference lines
    ax.axhline(ar_rmse, color=COLORS["AR(p)"], lw=1.5, ls="--",
               label=f"AR(p) = {ar_rmse:.2f}%", zorder=4)
    ax.axhline(rw_rmse, color=COLORS["RW"], lw=1.2, ls=":",
               label=f"RW = {rw_rmse:.2f}%", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in _KEY_MODELS])
    ax.set_ylabel("RMSE (%)")
    sample_label = "Pre-COVID" if sample == "pre_covid" else sample.replace("_", "-").title()
    ax.set_title(f"Figure 1 — RMSE by Model and Vintage ({sample_label} sample, 2005Q1–2019Q4)")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, max(ar_rmse, rw_rmse) * 1.55)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Time-series of GDP growth and key nowcasts
# ---------------------------------------------------------------------------

def fig2_time_series_nowcasts() -> plt.Figure:
    """
    Realized GDP growth with AR, best regression (bridge combo V3), and DFM V3.
    COVID period highlighted.  Large COVID outliers annotated.
    """
    realized_series = _load_forecast("benchmark_ar")["realized"]
    ar_fc = _load_forecast("benchmark_ar")["forecast"]
    bridge_fc = _load_forecast("bridge_combination_3")["forecast"]
    dfm_fc = _load_forecast("dfm_v3")["forecast"]

    # Convert to DatetimeIndex for cleaner x-axis
    def to_dt(s: pd.Series) -> pd.Series:
        return s.copy().rename(s.name).pipe(
            lambda x: x.set_axis(x.index.to_timestamp())
        )

    realized_dt = to_dt(realized_series)
    ar_dt = to_dt(ar_fc)
    bridge_dt = to_dt(bridge_fc)
    dfm_dt = to_dt(dfm_fc)

    fig, ax = plt.subplots(figsize=(12, 5))

    # COVID shading
    covid_s = pd.Period(COVID_START, "Q").to_timestamp()
    covid_e = pd.Period(COVID_END, "Q").to_timestamp()
    ax.axvspan(covid_s, covid_e, alpha=0.12, color="tomato", label="COVID (2020Q1–2021Q2)")

    # Clip y-axis; annotate the COVID outliers
    y_lo, y_hi = -15, 12
    ax.set_ylim(y_lo, y_hi)
    clipped = realized_dt[realized_dt < y_lo]
    for ts, val in clipped.items():
        ax.annotate(
            f"{val:.0f}%",
            xy=(ts, y_lo), xytext=(ts, y_lo + 1.5),
            fontsize=8, ha="center", color="tomato",
            arrowprops=dict(arrowstyle="-|>", color="tomato", lw=0.8),
        )

    # Lines
    ax.plot(realized_dt.index, realized_dt.values,
            color="black", lw=1.8, label="Realized GDP growth", zorder=5)
    ax.plot(ar_dt.index, ar_dt.values,
            color=COLORS["AR(p)"], lw=1.2, ls="--", label="AR(p)", zorder=3)
    ax.plot(bridge_dt.index, bridge_dt.values,
            color=COLORS["bridge_combination"], lw=1.4, label="Bridge combo (V3)", zorder=4)
    ax.plot(dfm_dt.index, dfm_dt.values,
            color=COLORS["dfm"], lw=1.2, ls="-.", label="DFM (V3)", zorder=3)

    ax.axhline(0, color="black", lw=0.5, ls="-", alpha=0.4)
    ax.set_ylabel("Annualized GDP growth (%)")
    ax.set_title(
        "Figure 2 — Realized vs. Nowcasted GDP Growth, Month-3 Vintage (2005Q1–present)\n"
        "(y-axis clipped at ±15%; 2020Q2 realized = −31.6%)"
    )
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: RMSE as function of within-quarter data arrival
# ---------------------------------------------------------------------------

def fig3_rmse_by_vintage(sample: str = "pre_covid") -> plt.Figure:
    """
    Line plot: RMSE vs. vintage for key models.
    Illustrates that accuracy improves as the quarter's data fills in.
    """
    summary = _load_summary()
    pre = summary[summary["sample"] == sample]

    # AR benchmark (no vintage) — horizontal line
    ar_rmse = float(pre.loc[pre["model"] == "AR(p)", "rmse"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Reference line
    ax.axhline(ar_rmse, color=COLORS["AR(p)"], lw=1.5, ls="--",
               label=f"AR(p) baseline ({ar_rmse:.2f}%)")

    vintages = [1, 2, 3]
    v_labels = ["Month 1", "Month 2", "Month 3"]

    for model in _KEY_MODELS:
        model_rows = pre[pre["model"] == model].dropna(subset=["vintage"])
        if model_rows.empty:
            continue
        rmses = [
            float(model_rows.loc[model_rows["vintage"] == v, "rmse"].values[0])
            if v in model_rows["vintage"].values else np.nan
            for v in vintages
        ]
        ax.plot(
            v_labels, rmses,
            marker="o", color=COLORS.get(model, "gray"),
            label=MODEL_LABELS.get(model, model),
            lw=1.6, ms=6,
        )

    ax.set_ylabel("RMSE (%)")
    sample_label = "Pre-COVID" if sample == "pre_covid" else sample.replace("_", "-").title()
    ax.set_title(
        f"Figure 3 — Forecast Accuracy Improves as the Quarter Fills In\n"
        f"({sample_label} sample, 2005Q1–2019Q4)"
    )
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Lasso selection stability
# ---------------------------------------------------------------------------

def fig4_lasso_selection_stability(top_n: int = 20) -> plt.Figure:
    """
    Horizontal bar chart: fraction of OOS quarters each feature was selected
    by Lasso (averaged across vintages), top-N features ranked by frequency.
    """
    sel = pd.read_csv(FORECASTS_DIR / "lasso_selection.csv")

    # Feature columns only (exclude metadata columns)
    meta_cols = {"quarter", "vintage", "best_alpha", "n_selected"}
    feat_cols = [c for c in sel.columns if c not in meta_cols]

    # Selection frequency per feature, averaged across all vintages
    freq = sel[feat_cols].apply(pd.to_numeric, errors="coerce").mean()
    freq_sorted = freq.sort_values(ascending=False).head(top_n)

    # Nicer feature labels: INDPRO_m1 → INDPRO (m1)
    labels = [f.replace("_m", " (m") + (")" if "_m" in f else "") for f in freq_sorted.index]

    # Color bars by indicator family
    bar_colors = []
    for feat in freq_sorted.index:
        sid = feat.split("_m")[0] if "_m" in feat else feat
        bar_colors.append(plt.cm.tab20(list(freq.index).index(feat) % 20))

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.33)))

    bars = ax.barh(
        range(len(freq_sorted)), freq_sorted.values * 100,
        color=bar_colors, edgecolor="white", lw=0.3,
    )
    ax.set_yticks(range(len(freq_sorted)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Selected in (%) of OOS quarters")
    ax.set_xlim(0, 105)
    ax.set_title(
        f"Figure 4 — Lasso Variable Selection Stability\n"
        f"(Top {top_n} features by selection frequency, averaged across vintages)"
    )
    # Value labels
    for bar, val in zip(bars, freq_sorted.values * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI: generate and save all four figures
# ---------------------------------------------------------------------------

def run_figures() -> None:
    print("Generating figures ...")

    print("  Figure 1: RMSE by model and vintage ...")
    save_fig(fig1_rmse_by_model_vintage(), "fig1_rmse_by_model_vintage")

    print("  Figure 2: Time-series of GDP growth and nowcasts ...")
    save_fig(fig2_time_series_nowcasts(), "fig2_time_series_nowcasts")

    print("  Figure 3: RMSE as function of data arrival ...")
    save_fig(fig3_rmse_by_vintage(), "fig3_rmse_by_vintage")

    print("  Figure 4: Lasso selection stability ...")
    save_fig(fig4_lasso_selection_stability(), "fig4_lasso_selection_stability")

    print(f"All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_figures()
