"""
GDP Nowcasting — Interactive Research Dashboard
Run with:  streamlit run dashboard.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COVID_END, COVID_START,
    DATA_PROCESSED_DIR, DATA_RAW_DIR,
    FORECASTS_DIR, TABLES_DIR,
    PREDICTOR_SERIES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GDP Nowcasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Dark / light mode  — injected CSS + Plotly template
# ─────────────────────────────────────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = False

DARK  = "#0f172a"
DARK2 = "#1e293b"
DARK3 = "#334155"
LIGHT_TEXT = "#f1f5f9"
MUTED = "#94a3b8"

LIGHT  = "#ffffff"
LIGHT2 = "#f1f5f9"
LIGHT3 = "#e2e8f0"
DARK_TEXT = "#1e293b"

def _css(dark: bool) -> str:
    bg   = DARK  if dark else LIGHT
    bg2  = DARK2 if dark else LIGHT2
    bg3  = DARK3 if dark else LIGHT3
    txt  = LIGHT_TEXT if dark else DARK_TEXT
    muted = MUTED if dark else "#64748b"
    border = DARK3 if dark else "#cbd5e1"
    return f"""
    <style>
    /* App background */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {bg};
        color: {txt};
    }}
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {bg2} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {txt} !important;
    }}
    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {bg2};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 16px 20px;
    }}
    [data-testid="stMetricValue"] {{
        color: {txt} !important;
        font-size: 1.8rem !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {muted} !important;
        font-size: 0.85rem !important;
    }}
    /* Headers */
    h1, h2, h3, h4 {{
        color: {txt} !important;
    }}
    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {bg2};
        border: 1px solid {border};
        border-radius: 8px;
    }}
    /* Tables */
    .stDataFrame, [data-testid="stTable"] {{
        background-color: {bg2};
    }}
    /* Tabs */
    [data-testid="stTabs"] button {{
        color: {muted} !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {txt} !important;
        border-bottom-color: #3b82f6 !important;
    }}
    /* Info / warning boxes */
    [data-testid="stInfo"] {{
        background-color: {bg2};
        border-left: 4px solid #3b82f6;
    }}
    /* Markdown text */
    .stMarkdown p, .stMarkdown li {{
        color: {txt};
    }}
    /* Top-bar */
    header[data-testid="stHeader"] {{
        background-color: {bg};
    }}
    /* Selectbox / multiselect */
    [data-baseweb="select"] {{
        background-color: {bg2} !important;
        color: {txt} !important;
    }}
    </style>
    """

def _plotly_tpl(dark: bool) -> str:
    return "plotly_dark" if dark else "plotly_white"

def _bg(dark: bool) -> str:
    return DARK if dark else LIGHT

def _paper(dark: bool) -> str:
    return DARK2 if dark else LIGHT

# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_monthly_panel() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "monthly_panel.parquet")
    df.index = pd.PeriodIndex(df.index, freq="M").to_timestamp()
    return df

@st.cache_data
def load_gdp_growth() -> pd.Series:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "quarterly_gdp.parquet")
    df.index = pd.PeriodIndex(df.index, freq="Q").to_timestamp()
    return df.iloc[:, 0].rename("gdp_growth")

@st.cache_data
def load_raw_series(sid: str) -> pd.Series:
    path = DATA_RAW_DIR / f"{sid}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    freq = "Q" if sid == "GDPC1" else "M"
    df.index = pd.PeriodIndex(df.index, freq=freq).to_timestamp()
    return df.iloc[:, 0].rename(sid)

@st.cache_data
def load_master_results() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "master_results.csv")

@st.cache_data
def load_forecast(name: str) -> pd.DataFrame:
    path = FORECASTS_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.PeriodIndex(df.index, freq="Q").to_timestamp()
    return df

@st.cache_data
def load_lasso_selection() -> pd.DataFrame:
    path = FORECASTS_DIR / "lasso_selection.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

# ─────────────────────────────────────────────────────────────────────────────
# Variable glossary
# ─────────────────────────────────────────────────────────────────────────────
GLOSSARY = [
    ("INDPRO",    "Industrial Production Index",
     "Measures how much physical output U.S. factories, mines, and utilities produce each month. "
     "A leading gauge of manufacturing health and economic momentum.",
     "Log-diff", "~17 days"),
    ("PAYEMS",    "Nonfarm Payrolls",
     "Total number of paid workers in the U.S. economy, excluding farm workers and self-employed. "
     "The single most-watched monthly jobs report, released the first Friday of each month.",
     "Log-diff", "~5 days"),
    ("RSAFS",     "Retail & Food Services Sales",
     "Total receipts at retail stores and food service establishments. Captures about 70% of "
     "consumer spending, which drives ~70% of GDP.",
     "Log-diff", "~14 days"),
    ("UNRATE",    "Unemployment Rate",
     "Percentage of the labor force that is actively seeking work but unemployed. "
     "Released alongside payrolls; moves slowly but signals labor-market slack.",
     "First-diff", "~5 days"),
    ("ICSA",      "Initial Jobless Claims",
     "Number of people filing for unemployment benefits for the first time in a week. "
     "Published every Thursday — one of the most timely economic indicators available.",
     "Log-diff", "~5 days"),
    ("HOUST",     "Housing Starts",
     "Number of new residential housing units on which construction has begun. "
     "Sensitive to interest rates; a leading indicator of construction employment and investment.",
     "Log-diff", "~19 days"),
    ("DGORDER",   "Durable Goods Orders",
     "Value of new orders placed with manufacturers for goods expected to last three or more years "
     "(machinery, aircraft, vehicles). Signals future production activity.",
     "Log-diff", "~28 days"),
    ("UMCSENT",   "Consumer Sentiment (U. of Michigan)",
     "Survey of households' views on personal finances and the broader economy. "
     "Forward-looking: pessimistic consumers tend to cut spending.",
     "First-diff", "0 days"),
    ("PCEPI",     "PCE Price Index",
     "The Federal Reserve's preferred inflation measure — broader than CPI and more "
     "representative of what Americans actually buy.",
     "Log-diff", "~28 days"),
    ("T10Y2Y",    "10Y–2Y Treasury Yield Spread",
     "The difference between 10-year and 2-year U.S. Treasury yields. "
     "An inverted yield curve (negative spread) has preceded every U.S. recession since 1955.",
     "First-diff", "~1 day"),
    ("BAA10Y",    "BAA–10Y Credit Spread",
     "Extra interest rate that corporations with BAA-rated (investment-grade) bonds pay "
     "over the risk-free 10-year Treasury rate. Widens in recessions as default risk rises.",
     "First-diff", "~1 day"),
    ("NASDAQCOM", "NASDAQ Composite Index",
     "Broad equity market index covering ~3,000 NASDAQ-listed stocks, heavily weighted toward "
     "technology. Serves as a real-time proxy for investor sentiment and financial conditions.",
     "Log-diff", "~1 day"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for models
# ─────────────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "AR(p)":             "#94a3b8",
    "RW":                "#cbd5e1",
    "bridge_combination":"#3b82f6",
    "midas_combination": "#f97316",
    "lasso":             "#22c55e",
    "elasticnet":        "#ef4444",
    "dfm":               "#a855f7",
    "combination":       "#f59e0b",
}
MODEL_LABELS = {
    "AR(p)":             "AR(p) baseline",
    "RW":                "Random walk",
    "bridge_combination":"Bridge combo",
    "midas_combination": "MIDAS combo",
    "lasso":             "Lasso",
    "elasticnet":        "ElasticNet",
    "dfm":               "DFM",
    "combination":       "Method combination",
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 📈 GDP Nowcasting")
        st.markdown("---")

        # Dark / light toggle
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("🌙" if st.session_state.dark else "☀️")
        with col2:
            toggled = st.toggle(
                "Dark mode",
                value=st.session_state.dark,
                key="dark_toggle",
                label_visibility="collapsed",
            )
        if toggled != st.session_state.dark:
            st.session_state.dark = toggled
            st.rerun()

        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["🏠  Overview", "📊  Data Explorer", "🔬  Model Comparison",
             "📈  Forecast History", "🔍  Insights"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#64748b'>Proteek Basu · 2026<br>"
            "Data: FRED · Models: OLS, Lasso,<br>ElasticNet, DFM · "
            "OOS: 2005–2026</small>",
            unsafe_allow_html=True,
        )

    return page

# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(dark: bool):
    tpl = _plotly_tpl(dark)

    st.markdown("# Nowcasting Quarterly U.S. Real GDP Growth")
    st.markdown(
        "**Can monthly economic data predict GDP before the official estimate is released?**  \n"
        "This project compares five forecasting approaches on 20 years of out-of-sample data "
        "and finds that smart combinations of monthly indicators can reduce forecast error by up to **30%** "
        "compared to a pure time-series baseline."
    )

    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AR Baseline RMSE", "2.47%", help="Pre-COVID root mean squared error of the AR(p) benchmark")
    c2.metric("Best Model RMSE", "1.72%", delta="-0.75 pp", help="Method combination, Month-3 vintage, pre-COVID")
    c3.metric("Error Reduction", "30%", help="RMSE improvement of best model over AR baseline")
    c4.metric("OOS Quarters", "85", help="Out-of-sample evaluation period: 2005Q1–2026Q1")
    c5.metric("Indicators Used", "12", help="Monthly FRED series used as predictors")

    st.markdown("---")

    # How it works
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("### The Challenge")
        st.markdown(
            """
GDP is reported **quarterly** — and the first official estimate arrives about six weeks after
the quarter ends. But monthly economic data streams in throughout the quarter: factory output,
payrolls, retail sales, jobless claims...

**Nowcasting** uses these early signals to predict the GDP number *before* it's officially
released. The trick is handling:

- **Mixed frequencies** — monthly indicators predicting a quarterly target
- **Publication delays** — each series has its own release lag after month-end
- **Three information vintages** — forecasts made at end of months 1, 2, and 3 of the quarter

The forecasts get more accurate as more data arrives during the quarter.
"""
        )

    with col_b:
        st.markdown("### Models Tested")
        for m, label in [
            ("AR(p)", "Pure GDP autoregression"),
            ("Bridge equations", "Per-indicator OLS"),
            ("MIDAS regression", "Monthly lags, unrestricted"),
            ("Lasso / ElasticNet", "Regularized, 37 features"),
            ("Dynamic Factor Model", "Kalman filter, latent factor"),
            ("Method combination", "Equal-weight ensemble"),
        ]:
            st.markdown(f"- **{m}** — {label}")

    st.markdown("---")
    st.markdown("### Headline Results (Pre-COVID, 2005Q1–2019Q4)")

    results = load_master_results()
    pre = results[results["sample"] == "pre_covid"].copy()
    key_models = ["AR(p)", "bridge_combination", "midas_combination",
                  "lasso", "elasticnet", "dfm", "combination"]
    pre_key = pre[pre["model"].isin(key_models)].copy()
    pre_key["vintage_label"] = pre_key["vintage"].apply(
        lambda v: f"Month {int(v)}" if str(v) not in ("all", "nan") and pd.notna(v) else "—"
    )
    pre_key["Model"] = pre_key["model"].map(MODEL_LABELS).fillna(pre_key["model"])
    pre_key["RMSE (%)"] = pre_key["rmse"].round(4)
    pre_key["Beats AR (10%)?"] = pre_key["dm_pval"].apply(
        lambda p: "✓ Yes" if (pd.notna(p) and p < 0.10) else ("— baseline" if pd.isna(p) else "✗ No")
    )
    pre_key["DM p-value"] = pre_key["dm_pval"].apply(
        lambda p: f"{p:.3f}" if pd.notna(p) else "—"
    )

    display = pre_key[["Model", "vintage_label", "RMSE (%)", "Beats AR (10%)?", "DM p-value"]]
    display = display.rename(columns={"vintage_label": "Vintage"})
    display = display.sort_values(["Vintage", "RMSE (%)"])
    st.dataframe(display, hide_index=True, use_container_width=True)

    st.info(
        "💡 **Key finding:** The method combination (equal-weight average of all four model families) "
        "achieves the lowest RMSE at all three vintages and is the only approach that is "
        "statistically significantly better than the AR baseline across all vintages (at the 10% level). "
        "No model clears the 5% threshold — standard with only 60 quarterly observations."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
def page_data_explorer(dark: bool):
    tpl = _plotly_tpl(dark)

    st.markdown("## Data Explorer")
    st.markdown(
        "Explore the 12 monthly predictors and quarterly GDP growth series. "
        "All series are shown in their **stationary, model-ready** form (differenced or log-differenced)."
    )

    # Glossary
    with st.expander("📖 Variable Glossary — click to expand", expanded=False):
        gdf = pd.DataFrame(
            GLOSSARY,
            columns=["Series ID", "Full Name", "What It Measures",
                     "Transform", "Publication Lag"],
        )
        st.dataframe(gdf, hide_index=True, use_container_width=True, height=420)

    st.markdown("---")

    # GDP growth chart
    st.markdown("### Quarterly GDP Growth (Target Variable)")
    gdp = load_gdp_growth().dropna()
    covid_s = pd.Period(COVID_START, "Q").to_timestamp()
    covid_e = pd.Period(COVID_END, "Q").to_timestamp()

    fig = go.Figure()
    fig.add_vrect(x0=covid_s, x1=covid_e, fillcolor="tomato",
                  opacity=0.12, line_width=0, annotation_text="COVID",
                  annotation_position="top left")
    fig.add_hline(y=0, line_color="gray", line_dash="dash", line_width=0.8)
    fig.add_trace(go.Scatter(
        x=gdp.index, y=gdp.values,
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(color="#3b82f6", width=2),
        name="GDP growth",
        hovertemplate="%{x|%YQ%q}<br><b>%{y:.2f}%</b> annualised<extra></extra>",
    ))
    fig.update_layout(
        template=tpl, height=320,
        yaxis_title="Annualised % growth",
        xaxis_title="",
        paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
        margin=dict(t=20, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Monthly Predictors")

    # Series selector
    all_sids = list(PREDICTOR_SERIES.keys())
    selected = st.multiselect(
        "Select indicators to display",
        options=all_sids,
        default=["INDPRO", "PAYEMS", "UNRATE", "T10Y2Y"],
        format_func=lambda s: f"{s} — {PREDICTOR_SERIES[s]['desc']}",
    )

    if not selected:
        st.info("Select at least one indicator above.")
        return

    panel = load_monthly_panel()

    # Date range
    min_date = panel.index.min().to_pydatetime()
    max_date = panel.index.max().to_pydatetime()
    import datetime
    date_range = st.slider(
        "Date range",
        min_value=min_date, max_value=max_date,
        value=(datetime.datetime(2000, 1, 1), max_date),
        format="YYYY-MM",
    )
    panel_filtered = panel.loc[date_range[0]:date_range[1]]

    colors = px.colors.qualitative.Plotly
    fig2 = go.Figure()
    for i, sid in enumerate(selected):
        if sid not in panel_filtered.columns:
            continue
        s = panel_filtered[sid].dropna()
        meta = PREDICTOR_SERIES[sid]
        fig2.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines",
            name=f"{sid}",
            line=dict(color=colors[i % len(colors)], width=1.5),
            hovertemplate=f"<b>{sid}</b><br>%{{x|%b %Y}}: %{{y:.4f}}<extra></extra>",
        ))

    fig2.update_layout(
        template=tpl, height=400,
        yaxis_title="Transformed value",
        xaxis_title="",
        paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
        margin=dict(t=20, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Definition card for selected series
    if len(selected) == 1:
        sid = selected[0]
        match = [row for row in GLOSSARY if row[0] == sid]
        if match:
            row = match[0]
            st.markdown(
                f"**{row[0]} — {row[1]}**  \n"
                f"{row[2]}  \n"
                f"*Transform:* {row[3]} | *Publication lag:* {row[4]} after reference month-end"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
def page_model_comparison(dark: bool):
    tpl = _plotly_tpl(dark)

    st.markdown("## Model Comparison")
    st.markdown(
        "Compare forecast accuracy (RMSE) across all models and vintages. "
        "Lower RMSE = better. The AR(p) baseline is the bar every model must beat."
    )

    results = load_master_results()

    # Sample selector
    sample_map = {
        "Pre-COVID (2005Q1–2019Q4)": "pre_covid",
        "Full OOS (2005Q1–2026Q1)": "full",
        "Ex-COVID (excl. 2020Q1–2021Q2)": "ex_covid",
    }
    sample_label = st.selectbox("Sample window", list(sample_map.keys()))
    sample = sample_map[sample_label]

    key_models = ["bridge_combination", "midas_combination",
                  "lasso", "elasticnet", "dfm", "combination"]

    sub = results[
        (results["sample"] == sample) &
        (results["model"].isin(key_models)) &
        results["vintage"].notna() &
        (results["vintage"].astype(str) != "all")
    ].copy()
    sub["vintage"] = sub["vintage"].astype(float).astype(int)
    sub["Model"] = sub["model"].map(MODEL_LABELS)
    sub["Vintage"] = sub["vintage"].map({1: "Month 1", 2: "Month 2", 3: "Month 3"})
    sub["color"] = sub["model"].map(MODEL_COLORS)

    # AR reference line
    ar_rmse = float(
        results[(results["model"] == "AR(p)") & (results["sample"] == sample)]["rmse"].iloc[0]
    )

    # ── Figure A: Grouped bar chart ──────────────────────────────────────────
    st.markdown("### RMSE by Model and Vintage")
    fig_bar = px.bar(
        sub, x="Model", y="rmse", color="Vintage",
        barmode="group",
        color_discrete_map={
            "Month 1": "#93c5fd", "Month 2": "#3b82f6", "Month 3": "#1d4ed8",
        },
        labels={"rmse": "RMSE (%)", "Model": ""},
        template=tpl,
        height=420,
        text=sub["rmse"].apply(lambda v: f"{v:.2f}"),
    )
    fig_bar.update_traces(textposition="outside", textfont_size=9)
    fig_bar.add_hline(
        y=ar_rmse, line_dash="dash", line_color="#94a3b8",
        annotation_text=f"AR(p) = {ar_rmse:.2f}%",
        annotation_position="top right",
    )
    fig_bar.update_layout(
        paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Figure B: RMSE vs vintage (information arrival) ──────────────────────
    st.markdown("### How Accuracy Improves as the Quarter Fills In")
    st.markdown(
        "_Each line shows how a model's RMSE falls as more monthly data arrives "
        "during the quarter. A steeper drop = more benefit from timely data._"
    )

    fig_line = go.Figure()
    fig_line.add_hline(
        y=ar_rmse, line_dash="dash", line_color="#94a3b8",
        annotation_text=f"AR(p) = {ar_rmse:.2f}%",
        annotation_position="top right",
    )
    for model_id in key_models:
        rows = sub[sub["model"] == model_id].sort_values("vintage")
        if rows.empty:
            continue
        fig_line.add_trace(go.Scatter(
            x=["Month 1", "Month 2", "Month 3"],
            y=rows["rmse"].values,
            mode="lines+markers",
            name=MODEL_LABELS.get(model_id, model_id),
            line=dict(color=MODEL_COLORS.get(model_id, "gray"), width=2),
            marker=dict(size=8),
            hovertemplate=f"<b>{MODEL_LABELS.get(model_id, model_id)}</b><br>%{{x}}: %{{y:.4f}}%<extra></extra>",
        ))
    fig_line.update_layout(
        template=tpl, height=380,
        yaxis_title="RMSE (%)",
        xaxis_title="Within-quarter vintage",
        paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── DM test table ─────────────────────────────────────────────────────────
    st.markdown("### Statistical Significance (Diebold-Mariano Test vs AR baseline)")
    st.markdown(
        "A positive DM statistic with a small p-value means the model significantly "
        "outperforms the AR(p) baseline. The threshold for significance is **p < 0.10** "
        "(10%) — the conventional bar with ~60 quarterly observations."
    )

    dm_display = sub[["Model", "Vintage", "rmse", "dm_stat", "dm_pval"]].copy()
    dm_display["RMSE (%)"] = dm_display["rmse"].round(4)
    dm_display["DM Statistic"] = dm_display["dm_stat"].round(3)
    dm_display["p-value"] = dm_display["dm_pval"].round(3)
    dm_display["Significant?"] = dm_display["dm_pval"].apply(
        lambda p: "✓ Yes (10%)" if pd.notna(p) and p < 0.10 else "✗ No"
    )
    st.dataframe(
        dm_display[["Model", "Vintage", "RMSE (%)", "DM Statistic", "p-value", "Significant?"]],
        hide_index=True, use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Forecast History
# ─────────────────────────────────────────────────────────────────────────────
def page_forecast_history(dark: bool):
    tpl = _plotly_tpl(dark)

    st.markdown("## Forecast History")
    st.markdown(
        "Compare model nowcasts against actual GDP growth over time. "
        "Select models and a vintage to see how each performed quarter by quarter."
    )

    # Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        model_options = {
            "AR(p) baseline":         ("benchmark_ar",         "ar"),
            "Bridge combination":     ("bridge_combination_3", "bridge"),
            "MIDAS combination":      ("midas_combination_3",  "midas"),
            "Lasso":                  ("lasso_v3",             "lasso"),
            "ElasticNet":             ("elasticnet_v3",        "enet"),
            "DFM":                    ("dfm_v3",               "dfm"),
            "Method combination":     ("combination_v3",       "combo"),
        }
        chosen = st.multiselect(
            "Select models to display",
            list(model_options.keys()),
            default=["AR(p) baseline", "Bridge combination", "Method combination"],
        )
    with col2:
        vintage = st.selectbox("Vintage", ["Month 3", "Month 2", "Month 1"])
        v_num = {"Month 1": 1, "Month 2": 2, "Month 3": 3}[vintage]

    # Build per-vintage filenames
    vintage_map = {
        "benchmark_ar":        "benchmark_ar",
        "bridge_combination_3": f"bridge_combination_{v_num}",
        "midas_combination_3":  f"midas_combination_{v_num}",
        "lasso_v3":             f"lasso_v{v_num}",
        "elasticnet_v3":        f"elasticnet_v{v_num}",
        "dfm_v3":               f"dfm_v{v_num}",
        "combination_v3":       f"combination_v{v_num}",
    }

    gdp = load_gdp_growth().dropna()
    covid_s = pd.Period(COVID_START, "Q").to_timestamp()
    covid_e = pd.Period(COVID_END, "Q").to_timestamp()

    fig = go.Figure()
    fig.add_vrect(x0=covid_s, x1=covid_e, fillcolor="tomato",
                  opacity=0.12, line_width=0, annotation_text="COVID",
                  annotation_position="top left")
    fig.add_hline(y=0, line_color="gray", line_dash="dot", line_width=0.6)

    # Realized GDP
    fig.add_trace(go.Scatter(
        x=gdp.index, y=gdp.values,
        mode="lines+markers",
        name="Realized GDP growth",
        line=dict(color="#1e293b" if not dark else "#f1f5f9", width=2.5),
        marker=dict(size=3),
        hovertemplate="Realized: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # Model forecasts
    model_color_map = {
        "AR(p) baseline":     "#94a3b8",
        "Bridge combination": "#3b82f6",
        "MIDAS combination":  "#f97316",
        "Lasso":              "#22c55e",
        "ElasticNet":         "#ef4444",
        "DFM":                "#a855f7",
        "Method combination": "#f59e0b",
    }
    for label in chosen:
        fname_key, _ = model_options[label]
        actual_fname = vintage_map.get(fname_key, fname_key)
        df = load_forecast(actual_fname)
        if df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df["forecast"].values,
            mode="lines",
            name=label,
            line=dict(color=model_color_map.get(label, "gray"), width=1.8, dash="dash"),
            hovertemplate=f"{label}: <b>%{{y:.2f}}%</b><extra></extra>",
        ))

    fig.update_layout(
        template=tpl, height=480,
        yaxis_title="Annualised GDP growth (%)",
        xaxis_title="",
        paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
        margin=dict(t=30, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[-15, 12]),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Y-axis clipped at ±15%. 2020Q2 realized GDP growth was −31.6% (off-chart).")

    # Scatter: predicted vs. actual
    st.markdown("---")
    st.markdown("### Predicted vs. Actual Scatter")
    st.markdown("A good model's points should cluster tightly along the 45° line.")

    scatter_model = st.selectbox(
        "Choose model for scatter",
        [m for m in chosen if m != "AR(p) baseline"] or ["Bridge combination"],
    )
    fname_key, _ = model_options.get(scatter_model, ("bridge_combination_3", "b"))
    actual_fname = vintage_map.get(fname_key, fname_key)
    df_sc = load_forecast(actual_fname)

    if not df_sc.empty:
        merged = df_sc[["forecast", "realized"]].dropna()
        lo = min(merged.min().min(), -2)
        hi = max(merged.max().max(), 4)
        fig_sc = px.scatter(
            merged, x="realized", y="forecast",
            labels={"realized": "Realized GDP growth (%)", "forecast": "Nowcast (%)"},
            template=tpl, height=380,
            color_discrete_sequence=[model_color_map.get(scatter_model, "#3b82f6")],
        )
        fig_sc.add_shape(
            type="line", x0=lo, y0=lo, x1=hi, y1=hi,
            line=dict(color="#94a3b8", dash="dash", width=1),
        )
        fig_sc.update_layout(
            paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Insights
# ─────────────────────────────────────────────────────────────────────────────
def page_insights(dark: bool):
    tpl = _plotly_tpl(dark)

    st.markdown("## Model Insights")

    # ── Lasso selection ───────────────────────────────────────────────────────
    st.markdown("### Which Variables Did Lasso Pick?")
    st.markdown(
        "At each expanding-window step, Lasso automatically zeroes out unimportant features. "
        "This chart shows what fraction of the time each feature had a non-zero coefficient, "
        "averaged across all three vintages. Higher = more consistently important."
    )

    sel = load_lasso_selection()
    if not sel.empty:
        meta_cols = {"quarter", "vintage", "best_alpha", "n_selected"}
        feat_cols = [c for c in sel.columns if c not in meta_cols]
        freq = sel[feat_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)

        top_n = st.slider("Show top N features", 10, 37, 20)
        freq_top = freq.head(top_n)

        # Nicer labels
        labels = [f.replace("_m", " — Month ") for f in freq_top.index]

        # Color by indicator family
        indicator_colors = {}
        palette = px.colors.qualitative.Safe
        for i, sid in enumerate(PREDICTOR_SERIES.keys()):
            indicator_colors[sid] = palette[i % len(palette)]
        bar_colors = [
            indicator_colors.get(f.split("_m")[0], "#94a3b8")
            for f in freq_top.index
        ]

        fig_sel = go.Figure(go.Bar(
            y=labels, x=freq_top.values * 100,
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="%{y}<br><b>%{x:.1f}%</b> of quarters selected<extra></extra>",
        ))
        fig_sel.update_yaxes(autorange="reversed")
        fig_sel.update_layout(
            template=tpl, height=max(350, top_n * 22),
            xaxis_title="Selected in (%) of OOS quarters",
            xaxis=dict(range=[0, 105]),
            paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
            margin=dict(t=20, b=40, l=10),
        )
        st.plotly_chart(fig_sel, use_container_width=True)

        st.markdown(
            "**Persistently selected (≥95% of quarters):** "
            "`INDPRO — Month 1`, `PAYEMS — Month 2`, `ICSA — Month 3`, `NASDAQCOM — Month 3`. "
            "These represent industrial production, employment, unemployment claims, and equity markets — "
            "the four pillars of the classic business cycle."
        )

    # ── DFM factor loadings ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Dynamic Factor Model — Factor Loadings")
    st.markdown(
        "The DFM extracts one hidden 'economic activity' factor from all 12 indicators. "
        "The loading tells you how strongly each indicator moves with the factor. "
        "Negative loadings: these series rise when the factor falls (and vice versa)."
    )

    loadings_path = TABLES_DIR / "dfm_factor_loadings.csv"
    if loadings_path.exists():
        loadings = pd.read_csv(loadings_path)
        loadings["series"] = loadings["param"].str.extract(r"->(.+)$")
        loadings = loadings.dropna(subset=["series"]).sort_values("loading")

        colors_load = ["#ef4444" if v < 0 else "#3b82f6" for v in loadings["loading"]]
        fig_load = go.Figure(go.Bar(
            y=loadings["series"], x=loadings["loading"],
            orientation="h",
            marker_color=colors_load,
            hovertemplate="%{y}<br>Loading: <b>%{x:.4f}</b><extra></extra>",
        ))
        fig_load.add_vline(x=0, line_color="gray", line_width=1)
        fig_load.update_yaxes(autorange="reversed")
        fig_load.update_layout(
            template=tpl, height=380,
            xaxis_title="Factor loading",
            paper_bgcolor=_paper(dark), plot_bgcolor=_paper(dark),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_load, use_container_width=True)
        st.markdown(
            "_Interpretation:_ PAYEMS and INDPRO load negatively → the factor rises when "
            "employment/production falls → this factor can be read as an **economic weakness** factor. "
            "UNRATE loads positively (unemployment rises with weakness). The DFM's positive GDP bias "
            "suggests the weakness factor may be too pessimistic on average."
        )

    # ── Key takeaways ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Key Takeaways")

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            "**What worked:**\n"
            "- Bridge and MIDAS combinations significantly beat the AR baseline (10% level)\n"
            "- Method combination is the overall winner — bias cancellation across models\n"
            "- Monthly data is most valuable at Month 2, once payrolls and IP are released\n"
            "- Lasso consistently selects INDPRO, PAYEMS, ICSA, NASDAQ — the classic business cycle"
        )
    with col2:
        st.warning(
            "**What didn't work as expected:**\n"
            "- DFM underperforms the AR baseline — publication lags mean Month 3 ≈ Month 2 informationally\n"
            "- No model significant at 5% — 60 quarterly observations is a hard constraint\n"
            "- COVID dummies didn't improve pre-COVID accuracy (by construction)\n"
            "- DFM requires a larger indicator panel (50–200 series) to show its full potential"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dark = st.session_state.dark
    st.markdown(_css(dark), unsafe_allow_html=True)

    page = render_sidebar()

    if "Overview" in page:
        page_overview(dark)
    elif "Data Explorer" in page:
        page_data_explorer(dark)
    elif "Model Comparison" in page:
        page_model_comparison(dark)
    elif "Forecast History" in page:
        page_forecast_history(dark)
    elif "Insights" in page:
        page_insights(dark)


if __name__ == "__main__":
    main()
