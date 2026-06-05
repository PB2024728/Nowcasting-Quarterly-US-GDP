"""
Generates explainer.html with embedded data and Plotly charts.
Run:  python build_explainer.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.config import (
    COVID_END, COVID_START,
    DATA_PROCESSED_DIR, FORECASTS_DIR, TABLES_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")

master = pd.read_csv(TABLES_DIR / "master_results.csv")

def load_fc(name):
    p = FORECASTS_DIR / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.PeriodIndex(df.index, freq="Q").to_timestamp()
    return df

ar      = load_fc("benchmark_ar")
bridge3 = load_fc("bridge_combination_3")
midas3  = load_fc("midas_combination_3")
combo3  = load_fc("combination_v3")
dfm3    = load_fc("dfm_v3")

sel_df = pd.read_csv(FORECASTS_DIR / "lasso_selection.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Build chart data as plain Python dicts → JSON
# ─────────────────────────────────────────────────────────────────────────────

## CHART 1 — RMSE grouped bar (pre-COVID)
KEY = ["bridge_combination","midas_combination","lasso","elasticnet","dfm","combination"]
LABELS = {
    "bridge_combination":"Bridge combo",
    "midas_combination":"MIDAS combo",
    "lasso":"Lasso",
    "elasticnet":"ElasticNet",
    "dfm":"DFM",
    "combination":"Method combination",
}
pre = master[(master["sample"]=="pre_covid") & master["model"].isin(KEY)].copy()
pre["vintage"] = pd.to_numeric(pre["vintage"], errors="coerce")
pre = pre.dropna(subset=["vintage"])
pre["vintage"] = pre["vintage"].astype(int)
ar_pre = float(master[(master["model"]=="AR(p)")&(master["sample"]=="pre_covid")]["rmse"].iloc[0])

chart1 = {
    "ar_rmse": round(ar_pre, 4),
    "traces": []
}
for v, color in [(1,"#93c5fd"),(2,"#3b82f6"),(3,"#1d4ed8")]:
    sub = pre[pre["vintage"]==v].set_index("model").reindex(KEY)
    chart1["traces"].append({
        "name": f"Month {v}",
        "x": [LABELS[m] for m in KEY],
        "y": [round(float(r),4) if pd.notna(r) else None for r in sub["rmse"]],
        "color": color,
    })

## CHART 2 — RMSE vs vintage (line)
chart2 = {"ar_rmse": round(ar_pre,4), "series": []}
COLORS2 = {
    "bridge_combination":"#3b82f6",
    "midas_combination":"#f97316",
    "lasso":"#22c55e",
    "elasticnet":"#ef4444",
    "dfm":"#a855f7",
    "combination":"#f59e0b",
}
for m in KEY:
    rows = pre[pre["model"]==m].sort_values("vintage")
    chart2["series"].append({
        "name": LABELS[m],
        "x": [f"Month {int(v)}" for v in rows["vintage"]],
        "y": [round(float(r),4) for r in rows["rmse"]],
        "color": COLORS2[m],
    })

## CHART 3 — GDP growth time series
def fmt_dates(df):
    return [d.strftime("%Y-%m-%d") for d in df.index]

# Align all to common OOS index
oos_start = pd.Timestamp("2005-01-01")
realized = ar["realized"].dropna()
realized = realized[realized.index >= oos_start]

def get_fc(df, col="forecast"):
    if df.empty: return [], []
    s = df[col].dropna()
    s = s[s.index >= oos_start]
    return fmt_dates(s.to_frame()), [round(float(v),4) for v in s.values]

r_dates = fmt_dates(realized.to_frame())
r_vals  = [round(float(v),4) for v in realized.values]

ar_d,  ar_v  = get_fc(ar)
br_d,  br_v  = get_fc(bridge3)
mi_d,  mi_v  = get_fc(midas3)
co_d,  co_v  = get_fc(combo3)
df_d,  df_v  = get_fc(dfm3)

covid_s = pd.Period(COVID_START,"Q").to_timestamp().strftime("%Y-%m-%d")
covid_e = pd.Period(COVID_END,"Q").to_timestamp().strftime("%Y-%m-%d")

chart3 = {
    "covid_start": covid_s, "covid_end": covid_e,
    "realized":     {"x": r_dates, "y": r_vals},
    "ar":           {"x": ar_d,   "y": ar_v},
    "bridge":       {"x": br_d,   "y": br_v},
    "midas":        {"x": mi_d,   "y": mi_v},
    "combination":  {"x": co_d,   "y": co_v},
    "dfm":          {"x": df_d,   "y": df_v},
}

## CHART 4 — Lasso selection stability
meta_cols = {"quarter","vintage","best_alpha","n_selected"}
feat_cols = [c for c in sel_df.columns if c not in meta_cols]
freq = sel_df[feat_cols].apply(pd.to_numeric, errors="coerce").mean()
freq_sorted = freq.sort_values(ascending=False).head(25)
feat_labels = [f.replace("_m"," – Month ") for f in freq_sorted.index]
# Color by indicator family (cycle through palette)
palette = ["#3b82f6","#f97316","#22c55e","#ef4444","#a855f7",
           "#f59e0b","#06b6d4","#ec4899","#84cc16","#14b8a6",
           "#6366f1","#f43f5e"]
sids = list({f.split("_m")[0] for f in feat_sorted.index} if False else
            list(dict.fromkeys(f.split("_m")[0] for f in freq_sorted.index)))
sid_color = {sid: palette[i % len(palette)] for i, sid in enumerate(sids)}
bar_colors = [sid_color.get(f.split("_m")[0] if "_m" in f else f, "#94a3b8")
              for f in freq_sorted.index]

chart4 = {
    "labels": feat_labels,
    "values": [round(float(v)*100, 1) for v in freq_sorted.values],
    "colors": bar_colors,
}

# ─────────────────────────────────────────────────────────────────────────────
# Serialise
# ─────────────────────────────────────────────────────────────────────────────
DATA_JS = f"""
const CHART1 = {json.dumps(chart1)};
const CHART2 = {json.dumps(chart2)};
const CHART3 = {json.dumps(chart3)};
const CHART4 = {json.dumps(chart4)};
"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Nowcasting U.S. GDP — Explainer</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    :root{
      --bg:#f0f4f8;--bg2:#e2e8f0;--card:#fff;--card2:#f8fafc;
      --text:#1e293b;--text2:#475569;--text3:#94a3b8;
      --accent:#2563eb;--accent-l:#dbeafe;--accent-m:#93c5fd;
      --border:#e2e8f0;--code-bg:#f1f5f9;--code-c:#0f172a;
      --ok-bg:#f0fdf4;--ok-bd:#bbf7d0;--ok-t:#166534;
      --warn-bg:#fffbeb;--warn-bd:#fde68a;--warn-t:#92400e;
      --info-bg:#eff6ff;--info-bd:#bfdbfe;
      --hero:linear-gradient(135deg,#1e3a5f,#1e40af 50%,#312e81);
      --sh:0 1px 3px rgba(0,0,0,.08),0 4px 12px rgba(0,0,0,.06);
      --sh2:0 10px 30px rgba(0,0,0,.10);
      --tr:all .25s cubic-bezier(.4,0,.2,1);
    }
    html.dark{
      --bg:#0b1120;--bg2:#111827;--card:#1e293b;--card2:#162032;
      --text:#f1f5f9;--text2:#94a3b8;--text3:#64748b;
      --accent:#60a5fa;--accent-l:#1e3a5f;--accent-m:#3b82f6;
      --border:#334155;--code-bg:#0f172a;--code-c:#93c5fd;
      --ok-bg:#052e16;--ok-bd:#166534;--ok-t:#4ade80;
      --warn-bg:#1c1205;--warn-bd:#78350f;--warn-t:#fbbf24;
      --info-bg:#1e3a5f;--info-bd:#1d4ed8;
      --hero:linear-gradient(135deg,#0f172a,#1e3a5f 50%,#1e1b4b);
      --sh:0 2px 8px rgba(0,0,0,.4);--sh2:0 12px 40px rgba(0,0,0,.5);
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    html{scroll-behavior:smooth;font-size:16px}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
         background:var(--bg);color:var(--text);line-height:1.75;transition:background .3s,color .3s}
    ::selection{background:var(--accent-l);color:var(--accent)}
    a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}

    /* NAV */
    nav{position:sticky;top:0;z-index:100;background:var(--card);border-bottom:1px solid var(--border);
        box-shadow:var(--sh);padding:0 2rem;display:flex;align-items:center;
        justify-content:space-between;height:56px;transition:background .3s}
    .nav-brand{font-weight:700;font-size:.95rem;color:var(--accent)}
    .nav-links{display:flex;gap:.2rem;list-style:none;overflow-x:auto;scrollbar-width:none}
    .nav-links::-webkit-scrollbar{display:none}
    .nav-links a{color:var(--text2);font-size:.78rem;font-weight:500;padding:.3rem .6rem;
                 border-radius:6px;transition:var(--tr);white-space:nowrap}
    .nav-links a:hover,.nav-links a.active{background:var(--accent-l);color:var(--accent)}
    #tbtn{background:var(--bg2);border:1px solid var(--border);border-radius:8px;cursor:pointer;
          padding:.35rem .7rem;font-size:.82rem;color:var(--text);transition:var(--tr);
          flex-shrink:0;margin-left:.5rem}
    #tbtn:hover{background:var(--accent-l);color:var(--accent)}

    /* HERO */
    .hero{background:var(--hero);padding:5rem 2rem 4rem;text-align:center;position:relative;overflow:hidden}
    .hero::before{content:"";position:absolute;inset:0;
      background:url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M20 20.5V18H0v5h5v5H0v5h20v-9.5zm-2 5.5h-1v-1h1v1zm0-4h-1v-1h1v1zm-4 4h-1v-1h1v1zm0-4h-1v-1h1v1z'/%3E%3C/g%3E%3C/svg%3E")}
    .hero-inner{position:relative;max-width:780px;margin:0 auto}
    .badge{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);
           color:#bfdbfe;font-size:.78rem;font-weight:600;padding:.25rem .85rem;border-radius:20px;
           margin-bottom:1.2rem;letter-spacing:.05em}
    .hero h1{color:#fff;font-size:clamp(1.8rem,5vw,2.8rem);font-weight:800;line-height:1.2;margin-bottom:.9rem}
    .hero h1 span{color:#93c5fd}
    .hero-sub{color:#bfdbfe;font-size:1.05rem;max-width:560px;margin:0 auto 1.75rem}
    .kpis{display:flex;gap:.85rem;justify-content:center;flex-wrap:wrap;margin-top:1.75rem}
    .kpi{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);border-radius:12px;
         padding:.85rem 1.2rem;text-align:center;min-width:120px}
    .kpi-v{font-size:1.75rem;font-weight:800;color:#fff;line-height:1}
    .kpi-l{font-size:.72rem;color:#93c5fd;margin-top:.25rem;font-weight:500}
    .kpi-d{font-size:.78rem;color:#86efac;margin-top:.15rem;font-weight:600}

    /* LAYOUT */
    main{max-width:920px;margin:0 auto;padding:0 1.5rem 5rem}
    section{padding-top:4rem}
    section+section{border-top:1px solid var(--border)}

    /* TYPOGRAPHY */
    h2{font-size:1.6rem;font-weight:800;color:var(--text);margin-bottom:.9rem;
       display:flex;align-items:center;gap:.5rem}
    h2 .ico{font-size:1.3rem}
    h3{font-size:1.05rem;font-weight:700;color:var(--text);margin:1.4rem 0 .45rem}
    p{color:var(--text2);margin-bottom:.8rem}
    p:last-child{margin-bottom:0}
    strong{color:var(--text);font-weight:700}
    em{color:var(--accent);font-style:normal;font-weight:600}
    ul,ol{padding-left:1.3rem;color:var(--text2);margin-bottom:.8rem}
    li{margin-bottom:.3rem}
    code{background:var(--code-bg);color:var(--code-c);font-family:"SF Mono",Consolas,monospace;
         font-size:.82em;padding:.12em .38em;border-radius:4px;border:1px solid var(--border)}
    pre{background:var(--code-bg);color:var(--code-c);font-family:"SF Mono",Consolas,monospace;
        font-size:.84rem;padding:1.1rem 1.4rem;border-radius:10px;border:1px solid var(--border);
        overflow-x:auto;margin:1rem 0;line-height:1.7}

    /* CARDS */
    .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.4rem;
          box-shadow:var(--sh);transition:var(--tr)}
    .card:hover{box-shadow:var(--sh2);transform:translateY(-1px)}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem;margin:1.1rem 0}
    .ch-card{background:var(--card);border:1px solid var(--border);border-left:4px solid var(--accent-m);
             border-radius:10px;padding:1rem 1.2rem;box-shadow:var(--sh)}
    .ch-num{display:inline-block;background:var(--accent-l);color:var(--accent);font-weight:800;
            font-size:.88rem;width:1.9rem;height:1.9rem;border-radius:50%;text-align:center;
            line-height:1.9rem;margin-bottom:.4rem}
    .ch-card h3{margin:0 0 .35rem;font-size:.95rem}
    .ch-card p{font-size:.88rem;margin:0}

    /* CALLOUTS */
    .callout{border-radius:10px;padding:1rem 1.3rem;margin:1.1rem 0;border-left:4px solid;font-size:.91rem}
    .ci{background:var(--info-bg);border-color:var(--info-bd);color:var(--text)}
    .cs{background:var(--ok-bg);border-color:var(--ok-bd);color:var(--ok-t)}
    .cw{background:var(--warn-bg);border-color:var(--warn-bd);color:var(--warn-t)}
    .ct{font-weight:800;margin-bottom:.3rem;display:flex;align-items:center;gap:.4rem}
    .callout p{margin:0;color:inherit}.callout ul{margin:.35rem 0 0 1.1rem}.callout li{margin-bottom:.2rem}

    /* TABLE */
    .tw{overflow-x:auto;border-radius:12px;border:1px solid var(--border);box-shadow:var(--sh);margin:1.1rem 0}
    table{width:100%;border-collapse:collapse;font-size:.88rem}
    thead{background:var(--accent)}
    thead th{color:#fff;font-weight:700;padding:.7rem .95rem;text-align:left;white-space:nowrap}
    tbody tr:nth-child(even){background:var(--card2)}
    tbody tr:hover{background:var(--accent-l)}
    td{padding:.6rem .95rem;color:var(--text2);border-bottom:1px solid var(--border);vertical-align:top}
    td:first-child{color:var(--text);font-weight:600;font-family:"SF Mono",Consolas,monospace;font-size:.8rem}
    .rt td:first-child{font-family:inherit;font-size:.88rem}
    .rt .best{color:#15803d;font-weight:700}
    html.dark .rt .best{color:#4ade80}
    .rt .sig::after{content:" ✓";color:#16a34a;font-weight:700}
    html.dark .rt .sig::after{color:#4ade80}
    .rt .combo{background:#fefce8;font-weight:700}
    html.dark .rt .combo{background:#1c1a05}
    .rt .base td{color:var(--text3)}

    /* TIMELINE */
    .tl{position:relative;padding-left:3rem;margin:1.4rem 0}
    .tl::before{content:"";position:absolute;left:.9rem;top:0;bottom:0;width:2px;background:var(--border)}
    .di{position:relative;margin-bottom:1.6rem}
    .di:last-child{margin-bottom:0}
    .dot{position:absolute;left:-2.85rem;top:.1rem;width:1.8rem;height:1.8rem;border-radius:50%;
         background:var(--accent);color:#fff;font-size:.62rem;font-weight:800;
         display:flex;align-items:center;justify-content:center;
         border:3px solid var(--bg);box-shadow:0 0 0 2px var(--accent)}
    .dtag{background:var(--accent-l);color:var(--accent);font-size:.7rem;font-weight:700;
          padding:.18rem .55rem;border-radius:20px}
    .dh{display:flex;align-items:center;gap:.55rem;margin-bottom:.35rem}
    .dtitle{font-weight:700;color:var(--text);font-size:.93rem}
    .db{color:var(--text2);font-size:.88rem;line-height:1.65}
    .dr{display:inline-block;margin-top:.45rem;background:var(--ok-bg);color:var(--ok-t);
        border:1px solid var(--ok-bd);border-radius:6px;padding:.22rem .7rem;font-size:.8rem;font-weight:600}
    .drw{background:var(--warn-bg);color:var(--warn-t);border-color:var(--warn-bd)}

    /* MODEL CARDS */
    .mc{background:var(--card);border:1px solid var(--border);border-radius:12px;
        padding:1.05rem 1.2rem;box-shadow:var(--sh)}
    .mn{font-weight:800;font-size:.93rem;color:var(--text);margin-bottom:.22rem}
    .md{font-size:.86rem;color:var(--text2);margin-bottom:.45rem}
    .ma{font-size:.8rem;color:var(--text3);font-style:italic;border-top:1px solid var(--border);
        padding-top:.38rem;margin-top:.38rem}
    .ma::before{content:"💡 ";font-style:normal}
    .winner{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-l)}
    .winner .mn{color:var(--accent)}

    /* CHART CONTAINERS */
    .chart-box{background:var(--card);border:1px solid var(--border);border-radius:14px;
               padding:1.2rem 1rem .5rem;box-shadow:var(--sh);margin:1.1rem 0}
    .chart-title{font-size:.93rem;font-weight:700;color:var(--text);padding:0 .4rem .6rem}
    .chart-sub{font-size:.78rem;color:var(--text3);padding:0 .4rem .4rem}

    /* BACK TOP */
    #bt{position:fixed;bottom:2rem;right:2rem;background:var(--accent);color:#fff;
        width:42px;height:42px;border-radius:50%;border:none;cursor:pointer;font-size:1.1rem;
        box-shadow:var(--sh2);opacity:0;transform:translateY(10px);
        transition:opacity .3s,transform .3s;z-index:99}
    #bt.vis{opacity:1;transform:translateY(0)}
    #bt:hover{background:#1d4ed8}

    footer{text-align:center;padding:1.75rem;font-size:.8rem;color:var(--text3);
           border-top:1px solid var(--border)}

    @media(max-width:640px){
      nav{padding:0 1rem}.nav-links{display:none}
      .hero{padding:3.5rem 1.2rem 3rem}main{padding:0 1rem 4rem}
      .kpis{gap:.5rem}.kpi{min-width:100px;padding:.65rem .9rem}.kpi-v{font-size:1.5rem}
    }
  </style>
</head>
<body>

<nav>
  <span class="nav-brand">📈 GDP Nowcasting</span>
  <ul class="nav-links">
    <li><a href="#idea">Hypothesis</a></li>
    <li><a href="#challenge">Challenge</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#timeline">Day by Day</a></li>
    <li><a href="#models">Models</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#charts">Charts</a></li>
    <li><a href="#limits">Limitations</a></li>
    <li><a href="#takeaway">Takeaway</a></li>
  </ul>
  <button id="tbtn" onclick="toggleTheme()">🌙 Dark</button>
</nav>

<div class="hero">
  <div class="hero-inner">
    <div class="badge">Pre-Masters Research Project · 2026</div>
    <h1>Nowcasting <span>U.S. GDP Growth</span><br>— Plain Language Explainer —</h1>
    <p class="hero-sub">Can monthly economic data predict GDP <strong style="color:#86efac">before</strong>
    the official estimate arrives? We tested five methods on 20 years of data to find out.</p>
    <div class="kpis">
      <div class="kpi"><div class="kpi-v">2.47%</div><div class="kpi-l">AR Baseline RMSE</div></div>
      <div class="kpi"><div class="kpi-v">1.72%</div><div class="kpi-l">Best Model RMSE</div><div class="kpi-d">↓ 30% better</div></div>
      <div class="kpi"><div class="kpi-v">85</div><div class="kpi-l">Quarters Tested</div></div>
      <div class="kpi"><div class="kpi-v">12</div><div class="kpi-l">Indicators Used</div></div>
      <div class="kpi"><div class="kpi-v">5</div><div class="kpi-l">Model Families</div></div>
    </div>
  </div>
</div>

<main>

<!-- HYPOTHESIS -->
<section id="idea">
  <h2><span class="ico">💡</span> The Hypothesis &amp; Motivation</h2>
  <p>Every three months, the U.S. government publishes an official estimate of how fast the economy grew. This number — <strong>GDP growth</strong> — is one of the most watched economic statistics in the world. The problem: it arrives about <strong>six weeks after the quarter ends</strong>, and the first estimate is subject to revision for years.</p>
  <div class="callout ci">
    <div class="ct">🎯 The Central Question</div>
    <p>Monthly economic indicators — factory output, job numbers, retail sales, stock prices — stream in <em>throughout</em> the quarter as it happens. <strong>Can we use these early signals to predict GDP before the official estimate is released?</strong> And if so, how much better are these predictions compared to simply extrapolating past GDP?</p>
  </div>
  <p>This is called <strong>"nowcasting"</strong> — a word combining "now" and "forecasting." Central banks (the Fed, ECB, Bank of England) use nowcasting to monitor the economy in real time before official statistics are published. This project rigorously tests five different nowcasting approaches on 20 years of data.</p>
</section>

<!-- CHALLENGE -->
<section id="challenge">
  <h2><span class="ico">⚡</span> Why This Is Hard</h2>
  <p>Four concrete challenges had to be solved explicitly:</p>
  <div class="grid">
    <div class="ch-card"><div class="ch-num">1</div><h3>Mixed Frequencies</h3><p>GDP is quarterly. Monthly indicators arrive throughout the quarter. A forecast made in early January uses different data than one in late March, both predicting the same Q1 number.</p></div>
    <div class="ch-card"><div class="ch-num">2</div><h3>Publication Delays</h3><p>Data doesn't arrive the moment a month ends. Payrolls come out ~5 days later. Factory output ~17 days. Durable goods ~28 days. We simulate exactly what a real forecaster would have seen on any given date.</p></div>
    <div class="ch-card"><div class="ch-num">3</div><h3>Too Few Data Points</h3><p>Only ~60 normal (pre-COVID) quarters to evaluate on. Statistical tests have limited power with this few observations — hard to prove one method is definitively better.</p></div>
    <div class="ch-card"><div class="ch-num">4</div><h3>The COVID Outlier</h3><p>2020Q2 GDP collapsed −31.6% annualized. No model trained on normal data could predict a pandemic. Results are reported three ways: full sample, pre-COVID, and COVID-excluded.</p></div>
  </div>
  <div class="callout cw">
    <div class="ct">⚠️ The "Ragged Edge" Problem</div>
    <p>At any point in time, data looks like a table with ragged right edges — recent months are only partially filled in. Every model was evaluated using <em>only data available on the actual forecast date</em> — never anything from the future.</p>
  </div>
</section>

<!-- DATA -->
<section id="data">
  <h2><span class="ico">📊</span> The Data</h2>
  <p>All data is from <a href="https://fred.stlouisfed.org" target="_blank">FRED</a> (Federal Reserve Economic Data), a free public database. The <strong>target</strong> is real GDP growth: <code>400 × log(GDP_t / GDP_{t-1})</code> — annualized quarterly percent change.</p>
  <h3>The 12 Monthly Predictors</h3>
  <div class="tw"><table>
    <thead><tr><th>Series</th><th>What It Measures</th><th>Transform</th><th>Pub. Lag</th></tr></thead>
    <tbody>
      <tr><td>INDPRO</td><td>Industrial production — factory, mine &amp; utility output</td><td>Log-diff</td><td>~17 days</td></tr>
      <tr><td>PAYEMS</td><td>Nonfarm payrolls — total workers on payrolls</td><td>Log-diff</td><td>~5 days</td></tr>
      <tr><td>RSAFS</td><td>Retail sales — consumer spending at stores</td><td>Log-diff</td><td>~14 days</td></tr>
      <tr><td>UNRATE</td><td>Unemployment rate</td><td>First-diff</td><td>~5 days</td></tr>
      <tr><td>ICSA</td><td>Initial jobless claims — new unemployment filers each week</td><td>Log-diff</td><td>~5 days</td></tr>
      <tr><td>HOUST</td><td>Housing starts — new homes beginning construction</td><td>Log-diff</td><td>~19 days</td></tr>
      <tr><td>DGORDER</td><td>Durable goods orders — orders for long-lasting goods</td><td>Log-diff</td><td>~28 days</td></tr>
      <tr><td>UMCSENT</td><td>Consumer sentiment — how households feel about the economy</td><td>First-diff</td><td>0 days</td></tr>
      <tr><td>PCEPI</td><td>PCE inflation — the Fed's preferred price measure</td><td>Log-diff</td><td>~28 days</td></tr>
      <tr><td>T10Y2Y</td><td>Yield curve spread — 10-year minus 2-year Treasury yields</td><td>First-diff</td><td>~1 day</td></tr>
      <tr><td>BAA10Y</td><td>Credit spread — corporate vs. government borrowing cost</td><td>First-diff</td><td>~1 day</td></tr>
      <tr><td>NASDAQCOM</td><td>NASDAQ composite — broad equity market proxy</td><td>Log-diff</td><td>~1 day</td></tr>
    </tbody>
  </table></div>
</section>

<!-- TIMELINE -->
<section id="timeline">
  <h2><span class="ico">📅</span> Day-by-Day: What We Built</h2>
  <div class="tl">
    <div class="di"><div class="dot">1</div><div class="dh"><span class="dtag">Day 1</span><span class="dtitle">Project Scaffolding</span></div><div class="db">Set up code structure, installed dependencies, created a central <code>config.py</code>, initialized git. Clean foundations for every day that followed.</div></div>
    <div class="di"><div class="dot">2</div><div class="dh"><span class="dtag">Day 2</span><span class="dtitle">Getting the Data</span></div><div class="db">Built a FRED data fetcher with parquet caching (re-runs take ~1 second from disk). Replaced the unavailable SP500 series with NASDAQCOM, which covers 1990–present.</div></div>
    <div class="di"><div class="dot">3</div><div class="dh"><span class="dtag">Day 3</span><span class="dtitle">The Ragged Edge</span></div><div class="db">Built the publication-lag masker — the most important infrastructure in the project. Sets cells to missing where the release hasn't happened yet. Defined three vintages (V1/V2/V3). 5 unit tests written and passing.</div></div>
    <div class="di"><div class="dot">4</div><div class="dh"><span class="dtag">Day 4</span><span class="dtitle">Benchmarks</span></div><div class="db">Built the AR(p) autoregression and random-walk benchmarks using an expanding-window backtest from 2005Q1 — never peeking at the future.<div class="dr">📊 AR(p) pre-COVID RMSE = 2.47% — the bar to beat</div></div></div>
    <div class="di"><div class="dot">5</div><div class="dh"><span class="dtag">Day 5</span><span class="dtitle">Bridge Equations</span></div><div class="db">Per-indicator OLS regressions, combined by averaging. Partial aggregation handles ragged edge.<div class="dr">📊 Bridge V3 RMSE = 1.87% — beats AR by 24%</div></div></div>
    <div class="di"><div class="dot">6</div><div class="dh"><span class="dtag">Day 6</span><span class="dtitle">MIDAS Regression</span></div><div class="db">Keeps three monthly observations as separate features (not collapsed to quarterly average). Better at early vintages when granularity matters most.<div class="dr">📊 MIDAS V1 RMSE = 2.11% — beats bridge at earliest vintage</div></div></div>
    <div class="di"><div class="dot">7</div><div class="dh"><span class="dtag">Day 7</span><span class="dtitle">Regularized Regression</span></div><div class="db">All 12 indicators × 3 monthly lags = 37 features in one model. Lasso auto-selects: INDPRO, PAYEMS, ICSA, NASDAQ chosen ≥95% of the time.<div class="dr">📊 ElasticNet V3 RMSE = 1.90%</div></div></div>
    <div class="di"><div class="dot">8</div><div class="dh"><span class="dtag">Day 8</span><span class="dtitle">Dynamic Factor Model</span></div><div class="db">Kalman filter extracts a hidden "economic activity" factor from all 12 series simultaneously. Handles missing data natively. But publication lags limit what's actually available at quarter-end.<div class="dr drw">⚠️ DFM V3 RMSE = 2.77% — worse than AR baseline</div></div></div>
    <div class="di"><div class="dot">9</div><div class="dh"><span class="dtag">Day 9</span><span class="dtitle">Statistical Testing</span></div><div class="db">Implemented the Diebold-Mariano test with HAC standard errors. Ran evaluation across three sample windows. Tested COVID dummies in regularized models.</div></div>
    <div class="di"><div class="dot">10</div><div class="dh"><span class="dtag">Day 10</span><span class="dtitle">Publication Figures</span></div><div class="db">Four figures: RMSE bar chart, GDP time series, RMSE vs. vintage, Lasso selection map. PNG (300 DPI) + PDF.</div></div>
    <div class="di"><div class="dot">11</div><div class="dh"><span class="dtag">Day 11</span><span class="dtitle">Report &amp; Documentation</span></div><div class="db">Full technical report, updated README with step-by-step reproduction instructions.</div></div>
    <div class="di"><div class="dot">12</div><div class="dh"><span class="dtag">Day 12</span><span class="dtitle">Combination Forecast &amp; Polish</span></div><div class="db">Equal-weight average of Bridge + MIDAS + ElasticNet + DFM. Bias cancellation makes this the best model overall. 19 tests passing. Tagged v1.0.<div class="dr">🏆 Combination V3 RMSE = 1.72% — overall winner</div></div></div>
  </div>
</section>

<!-- MODELS -->
<section id="models">
  <h2><span class="ico">🔬</span> The Models Explained Simply</h2>
  <div class="grid">
    <div class="mc"><div class="mn">AR(p) — Autoregression</div><div class="md">Uses only past GDP to predict future GDP. Automatically picks how many past quarters to include. The benchmark.</div><div class="ma">Predicting tomorrow's weather using only today's temperature — no other data allowed.</div></div>
    <div class="mc"><div class="mn">Bridge Equations</div><div class="md">Each of the 12 indicators gets its own simple regression against GDP. The 12 individual forecasts are averaged together.</div><div class="ma">12 expert analysts each read one indicator and give a GDP estimate. Take the average of all 12 opinions.</div></div>
    <div class="mc"><div class="mn">MIDAS Regression</div><div class="md">Like bridge, but keeps monthly observations separate rather than averaging them. Lets the model weight months differently.</div><div class="ma">Same 12 experts, but each one can split their view into early, middle, and late-month observations.</div></div>
    <div class="mc"><div class="mn">Lasso / ElasticNet</div><div class="md">All 12 indicators × 3 monthly lags = 37 features in one model. Regularization auto-zeros unimportant features. Time-series cross-validation.</div><div class="ma">One expert reads all 37 data points but has a strict rule: ignore everything that doesn't genuinely help.</div></div>
    <div class="mc"><div class="mn">Dynamic Factor Model</div><div class="md">Assumes a hidden "economic health" factor drives all 12 indicators. Kalman filter estimates this factor and extracts a GDP nowcast.</div><div class="ma">A doctor diagnosing overall health from 12 vital signs — looking for the underlying condition, not reading each sign separately.</div></div>
    <div class="mc winner"><div class="mn">🏆 Method Combination</div><div class="md">Simple equal-weight average of Bridge, MIDAS, ElasticNet, and DFM. No single model dominates. Biases from different models partially cancel out.</div><div class="ma">Committee vote — every method family gets one equal vote, and errors offset each other.</div></div>
  </div>
</section>

<!-- RESULTS TABLE -->
<section id="results">
  <h2><span class="ico">📈</span> The Results</h2>
  <h3>Headline RMSE Table (Pre-COVID, 2005Q1–2019Q4)</h3>
  <p><em>Numbers are RMSE in annualised percent. Lower = better. ✓ = significantly better than AR at 10% level.</em></p>
  <div class="tw"><table class="rt">
    <thead><tr><th>Model</th><th>Month 1</th><th>Month 2</th><th>Month 3</th></tr></thead>
    <tbody>
      <tr class="base"><td>AR(p) — the bar to beat</td><td>2.47%</td><td>2.47%</td><td>2.47%</td></tr>
      <tr><td>Bridge combination</td><td>2.22%</td><td class="sig">1.89%</td><td class="sig">1.87%</td></tr>
      <tr><td>MIDAS combination</td><td class="sig">2.11%</td><td class="sig">1.93%</td><td class="sig">1.93%</td></tr>
      <tr><td>Lasso</td><td>2.36%</td><td>2.17%</td><td>2.07%</td></tr>
      <tr><td>ElasticNet</td><td>2.31%</td><td>2.06%</td><td>1.90%</td></tr>
      <tr><td>DFM</td><td>2.85%</td><td>2.79%</td><td>2.77%</td></tr>
      <tr class="combo"><td>🏆 Method combination</td><td class="sig best">1.90%</td><td class="sig best">1.77%</td><td class="sig best">1.72%</td></tr>
    </tbody>
  </table></div>
  <div class="callout cs"><div class="ct">✅ What the 30% Improvement Means in Practice</div>
    <p>If GDP is going to grow at 2.5%, the AR model's forecast could land anywhere from 0% to 5%. The combination model narrows that to roughly 0.8% to 4.2% — a meaningfully tighter interval for decision-makers.</p>
  </div>
</section>

<!-- CHARTS -->
<section id="charts">
  <h2><span class="ico">📊</span> Interactive Charts</h2>
  <p>Hover over any chart for exact values. Click legend entries to show or hide individual series.</p>

  <!-- Chart 1: RMSE bar -->
  <div class="chart-box">
    <div class="chart-title">Chart 1 — RMSE by Model and Vintage (Pre-COVID)</div>
    <div class="chart-sub">Each group of bars is one model; the three shades show Month 1 / 2 / 3 vintages. Dashed line = AR(p) baseline.</div>
    <div id="c1" style="height:420px"></div>
  </div>

  <!-- Chart 2: RMSE vs vintage -->
  <div class="chart-box">
    <div class="chart-title">Chart 2 — How Accuracy Improves as the Quarter Fills In</div>
    <div class="chart-sub">Each line traces one model's RMSE across the three within-quarter vintages. A steeper drop means the model benefits more from new data.</div>
    <div id="c2" style="height:380px"></div>
  </div>

  <!-- Chart 3: Time series -->
  <div class="chart-box">
    <div class="chart-title">Chart 3 — Realized GDP Growth vs. Key Model Nowcasts (Month-3 Vintage)</div>
    <div class="chart-sub">Solid black = realized GDP growth. Dashed lines = nowcasts. Red shading = COVID period. Y-axis clipped at ±15% (2020Q2 was −31.6%).</div>
    <div id="c3" style="height:440px"></div>
  </div>

  <!-- Chart 4: Lasso selection -->
  <div class="chart-box">
    <div class="chart-title">Chart 4 — Lasso Variable Selection Stability</div>
    <div class="chart-sub">Fraction of OOS quarters (averaged across all vintages) in which each feature had a non-zero Lasso coefficient. Higher = more consistently important.</div>
    <div id="c4" style="height:520px"></div>
  </div>

  <h3>Key Chart Insights</h3>
  <div class="callout cw"><div class="ct">⚠️ Why the DFM Underperforms</div>
    <p>The DFM (purple line, Chart 2) barely improves from Month 2 to Month 3 — and actually sits above the AR baseline throughout. The reason: at quarter-end, most month-3 data still hasn't been published (publication lags extend into the following month). The Kalman filter has little new information to work with, so V3 ≈ V2 informationally. A DFM needs 50–200 indicators to show its full potential.</p>
  </div>
  <div class="callout cs"><div class="ct">✅ Why the Combination (orange, Chart 2) Wins</div>
    <p>The regression models (Bridge, MIDAS) consistently under-predict GDP. The DFM consistently over-predicts. When averaged, these biases cancel out. The combination also reduces variance through diversification — the same logic as a balanced investment portfolio.</p>
  </div>
</section>

<!-- LIMITATIONS -->
<section id="limits">
  <h2><span class="ico">🔎</span> Key Limitations</h2>
  <div class="callout cw" style="margin-bottom:.9rem"><div class="ct">⚠️ Revised Data, Not Real-Time Data</div><p>The FRED data used is the <em>current, most recently revised</em> vintage. A forecaster in 2005 would have seen noisier data — revisions hadn't happened yet. Our accuracy figures are probably slightly optimistic. A fully rigorous study would use ALFRED (Archival FRED) for exact historical vintages.</p></div>
  <div class="callout cw" style="margin-bottom:.9rem"><div class="ct">⚠️ Approximate Publication Lags</div><p>Publication delays are fixed constants (e.g., "payrolls = 5 days after month-end"). In reality they shift by a few days each month. A production system would track the actual release calendar in real time.</p></div>
  <div class="callout cw" style="margin-bottom:.9rem"><div class="ct">⚠️ No Professional Forecaster Comparison</div><p>The NY Fed publishes a weekly nowcast. The Survey of Professional Forecasters publishes quarterly consensus estimates. We don't know if our methods add value beyond what professional forecasters already produce with far larger datasets.</p></div>
  <div class="callout cw"><div class="ct">⚠️ Small Panel for the DFM</div><p>A production DFM at a central bank would use 100+ series. With 12 indicators, the DFM can't show its full potential. With FRED-MD's ~120 series, results would likely be much better.</p></div>
</section>

<!-- TAKEAWAY -->
<section id="takeaway">
  <h2><span class="ico">🎯</span> The Big Takeaway</h2>
  <div class="callout cs" style="font-size:1rem"><div class="ct" style="font-size:1.05rem">✅ The Answer</div>
    <p><strong>Yes — monthly economic data does help predict GDP before the official estimate arrives.</strong> But the improvement is modest, and mainly materialises in the <em>middle of the quarter</em> once payrolls and industrial production have been released.</p>
  </div>
  <p>The best result — a <strong>30% reduction in forecast error</strong> — is economically meaningful but statistically fragile with only 60 quarterly observations. The combination of methods adds genuine value through bias diversification.</p>
  <p>Surprisingly, simple methods (Bridge, MIDAS) match or beat more sophisticated ones (Lasso, DFM) at this dataset size. Complexity doesn't always win.</p>
  <div class="card" style="text-align:center;padding:2rem;margin-top:1.4rem">
    <div style="font-size:2.5rem;margin-bottom:.6rem">🏆</div>
    <div style="font-size:1.2rem;font-weight:800;color:var(--accent);margin-bottom:.45rem">If you had to pick one model for live nowcasting with 12 indicators...</div>
    <div style="font-size:.97rem;color:var(--text2)">Use the <strong>method combination</strong> — equal-weight average of Bridge, MIDAS, ElasticNet, and DFM.<br>Pre-COVID RMSE = <strong>1.72%</strong> at Month-3 vintage, significant at the 10% level for all three vintages.</div>
  </div>
</section>

</main>

<footer>
  <p><strong>Proteek Basu</strong> · Pre-Masters Project #3 · June 2026</p>
  <p style="margin-top:.35rem">Data: <a href="https://fred.stlouisfed.org" target="_blank">FRED (St. Louis Fed)</a> · OOS: 2005Q1–2026Q1 · Git tag: <code>v1.0</code></p>
</footer>

<button id="bt" onclick="window.scrollTo({top:0,behavior:'smooth'})" title="Back to top">↑</button>

<script>
%%DATA_JS%%

/* ── Theme ── */
const tbtn = document.getElementById('tbtn');
function applyTheme(dark) {
  document.documentElement.classList.toggle('dark', dark);
  tbtn.textContent = dark ? '☀️ Light' : '🌙 Dark';
}
function toggleTheme() {
  const d = !document.documentElement.classList.contains('dark');
  localStorage.setItem('theme', d ? 'dark' : 'light');
  applyTheme(d);
  renderCharts(); // re-render with correct template
}
(function(){
  const s = localStorage.getItem('theme');
  const p = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(s ? s==='dark' : p);
})();

/* ── Back to top ── */
const bt = document.getElementById('bt');
window.addEventListener('scroll', () => bt.classList.toggle('vis', scrollY > 400), {passive:true});

/* ── Nav active ── */
const secs = document.querySelectorAll('section[id]');
const als  = document.querySelectorAll('.nav-links a');
new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      als.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href')==='#'+e.target.id);
      });
    }
  });
}, {rootMargin:'-20% 0px -70% 0px'}).observe(secs[0] || document.body);
secs.forEach(s => {
  new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting)
        als.forEach(a => a.classList.toggle('active', a.getAttribute('href')==='#'+e.target.id));
    });
  }, {rootMargin:'-20% 0px -70% 0px'}).observe(s);
});

/* ── Plotly helpers ── */
function isDark() { return document.documentElement.classList.contains('dark'); }
function tpl()    { return isDark() ? 'plotly_dark' : 'plotly_white'; }
function bg()     { return isDark() ? '#1e293b' : '#ffffff'; }
function gridC()  { return isDark() ? '#334155' : '#e2e8f0'; }
function txtC()   { return isDark() ? '#f1f5f9' : '#1e293b'; }

const LAY = (extra={}) => ({
  template: tpl(), paper_bgcolor: bg(), plot_bgcolor: bg(),
  font: {color: txtC(), family: '-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif', size: 12},
  xaxis: {gridcolor: gridC(), linecolor: gridC(), zerolinecolor: gridC()},
  yaxis: {gridcolor: gridC(), linecolor: gridC(), zerolinecolor: gridC()},
  legend: {bgcolor: 'rgba(0,0,0,0)', bordercolor: 'rgba(0,0,0,0)'},
  margin: {t:20, b:50, l:60, r:20},
  hovermode: 'x unified',
  ...extra
});

function renderCharts() {

  /* Chart 1 — RMSE grouped bar */
  (function(){
    const traces = CHART1.traces.map(t => ({
      type:'bar', name:t.name, x:t.x, y:t.y,
      marker:{color:t.color},
      text: t.y.map(v => v!=null ? v.toFixed(2) : ''),
      textposition:'outside', textfont:{size:9},
      hovertemplate:'<b>%{x}</b><br>RMSE: %{y:.4f}%<extra>'+t.name+'</extra>'
    }));
    const layout = LAY({
      barmode:'group',
      yaxis:{...LAY().yaxis, title:'RMSE (%)', range:[0, 3.5]},
      shapes:[{type:'line', x0:-0.5, x1:5.5, y0:CHART1.ar_rmse, y1:CHART1.ar_rmse,
               line:{color:'#94a3b8',dash:'dash',width:1.5}}],
      annotations:[{x:5.5, y:CHART1.ar_rmse, text:'AR(p) '+CHART1.ar_rmse+'%',
                    showarrow:false, xanchor:'right', yanchor:'bottom',
                    font:{color:'#94a3b8',size:11}}],
      legend:{orientation:'h', y:1.08}
    });
    Plotly.newPlot('c1', traces, layout, {responsive:true, displayModeBar:false});
  })();

  /* Chart 2 — RMSE vs vintage line */
  (function(){
    const traces = CHART2.series.map(s => ({
      type:'scatter', mode:'lines+markers', name:s.name, x:s.x, y:s.y,
      line:{color:s.color, width:2.2}, marker:{size:8, color:s.color},
      hovertemplate:'<b>'+s.name+'</b><br>%{x}: %{y:.4f}%<extra></extra>'
    }));
    traces.push({
      type:'scatter', mode:'lines', name:'AR(p) baseline',
      x:['Month 1','Month 2','Month 3'],
      y:[CHART2.ar_rmse,CHART2.ar_rmse,CHART2.ar_rmse],
      line:{color:'#94a3b8',dash:'dash',width:1.5},
      hoverinfo:'skip'
    });
    const layout = LAY({
      yaxis:{...LAY().yaxis, title:'RMSE (%)', range:[1.4, 3.1]},
      xaxis:{...LAY().xaxis, title:'Within-quarter vintage'},
      legend:{orientation:'h', y:1.08}
    });
    Plotly.newPlot('c2', traces, layout, {responsive:true, displayModeBar:false});
  })();

  /* Chart 3 — Time series */
  (function(){
    const c3 = CHART3;
    const traces = [
      {type:'scatter', mode:'lines+markers', name:'Realized GDP growth',
       x:c3.realized.x, y:c3.realized.y,
       line:{color: isDark()?'#f1f5f9':'#1e293b', width:2.5},
       marker:{size:3},
       hovertemplate:'Realized: <b>%{y:.2f}%</b><extra></extra>'},
      {type:'scatter', mode:'lines', name:'AR(p) baseline',
       x:c3.ar.x, y:c3.ar.y,
       line:{color:'#94a3b8',dash:'dash',width:1.5},
       hovertemplate:'AR(p): <b>%{y:.2f}%</b><extra></extra>'},
      {type:'scatter', mode:'lines', name:'Bridge combo',
       x:c3.bridge.x, y:c3.bridge.y,
       line:{color:'#3b82f6',dash:'dot',width:1.8},
       hovertemplate:'Bridge: <b>%{y:.2f}%</b><extra></extra>'},
      {type:'scatter', mode:'lines', name:'MIDAS combo',
       x:c3.midas.x, y:c3.midas.y,
       line:{color:'#f97316',dash:'dot',width:1.8},
       hovertemplate:'MIDAS: <b>%{y:.2f}%</b><extra></extra>'},
      {type:'scatter', mode:'lines', name:'Method combination',
       x:c3.combination.x, y:c3.combination.y,
       line:{color:'#f59e0b',dash:'dash',width:2.2},
       hovertemplate:'Combination: <b>%{y:.2f}%</b><extra></extra>'},
      {type:'scatter', mode:'lines', name:'DFM',
       x:c3.dfm.x, y:c3.dfm.y,
       line:{color:'#a855f7',dash:'dot',width:1.6},
       hovertemplate:'DFM: <b>%{y:.2f}%</b><extra></extra>'},
    ];
    const layout = LAY({
      yaxis:{...LAY().yaxis, title:'Annualised GDP growth (%)', range:[-15,12]},
      xaxis:{...LAY().xaxis, title:''},
      shapes:[{type:'rect', x0:c3.covid_start, x1:c3.covid_end,
               y0:-15, y1:12, fillcolor:'rgba(239,68,68,0.10)',
               line:{width:0}}],
      annotations:[{x:c3.covid_start, y:11, text:'COVID', showarrow:false,
                    font:{color:'#ef4444',size:11}}],
      legend:{orientation:'h', y:1.06}
    });
    Plotly.newPlot('c3', traces, layout, {responsive:true, displayModeBar:false});
  })();

  /* Chart 4 — Lasso selection */
  (function(){
    const trace = {
      type:'bar', orientation:'h',
      x: CHART4.values,
      y: CHART4.labels,
      marker:{color: CHART4.colors},
      text: CHART4.values.map(v => v.toFixed(1)+'%'),
      textposition:'outside',
      textfont:{size:9},
      hovertemplate:'<b>%{y}</b><br>Selected: %{x:.1f}% of quarters<extra></extra>'
    };
    const layout = LAY({
      yaxis:{...LAY().yaxis, autorange:'reversed', title:'', tickfont:{size:10}},
      xaxis:{...LAY().xaxis, title:'Selected in (%) of OOS quarters', range:[0,112]},
      margin:{t:20, b:50, l:170, r:50},
      hovermode:'y unified'
    });
    Plotly.newPlot('c4', [trace], layout, {responsive:true, displayModeBar:false});
  })();
}

// Initial render
document.addEventListener('DOMContentLoaded', renderCharts);
</script>
</body>
</html>
"""

# Inject data
HTML_FINAL = HTML.replace("%%DATA_JS%%", DATA_JS)

out = ROOT / "explainer.html"
out.write_text(HTML_FINAL, encoding="utf-8")
print(f"Written: {out}  ({out.stat().st_size/1024:.0f} KB)")
