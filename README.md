# Nowcasting Quarterly U.S. Real GDP Growth

Investigates whether monthly macroeconomic indicators can nowcast quarterly U.S. real GDP growth more accurately than a pure time-series benchmark. Four method families are compared — bridge equations, MIDAS regression, regularized regression (Lasso / ElasticNet), and a dynamic factor model — against AR(p) and random-walk baselines over a 2005Q1–2026Q1 out-of-sample window.

See [`Dashboard/report.md`](Dashboard/report.md) for the full write-up including results and limitations.

---

## Headline Results (Pre-COVID RMSE, 2005Q1–2019Q4)

| Model | Month 1 | Month 2 | Month 3 |
|---|---|---|---|
| AR(p) baseline | 2.47% | 2.47% | 2.47% |
| Bridge combination | 2.22% | 1.89% ✓ | 1.87% ✓ |
| MIDAS combination | 2.11% ✓ | 1.93% ✓ | 1.93% ✓ |
| ElasticNet | 2.31% | 2.06% | 1.90% |
| Lasso | 2.36% | 2.17% | 2.07% |
| DFM | 2.85% | 2.79% | 2.77% |
| **Method combination** | **1.90% ✓** | **1.77% ✓** | **1.72% ✓** |

✓ = significantly better than AR(p) at the 10% level (Diebold-Mariano test, HAC SE).  
No model reaches the 5% threshold with 60 quarterly observations.  
The **method combination** (equal-weight average of Bridge, MIDAS, ElasticNet, DFM) is the overall best model.

---

## Reproduction

### Prerequisites

- Python 3.11+ (tested on 3.14)
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
- ~200 MB disk space for data cache and results

### Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd "Nowcasting Quarterly US GDP (Project #3)"

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your FRED API key
cp .env.example .env
# Edit .env and set: FRED_API_KEY=your_key_here
```

### Run the pipeline

Run each step in order. Each step reads from the previous step's saved outputs, so they can be re-run individually if needed.

```bash
# Day 2: Download and cache all 13 FRED series
python -m src.data.fetch

# Day 3: Apply stationarity transforms, save processed panels
python -m src.data.transforms

# Day 4: AR(p) and random-walk benchmarks
python -m src.models.benchmarks

# Day 5: Bridge equations (all indicators × 3 vintages)
python -m src.models.bridge

# Day 6: U-MIDAS regression (all indicators × 3 vintages)
python -m src.models.midas

# Day 7: Lasso and ElasticNet (multi-indicator, 3 vintages)
python -m src.models.regularized

# Day 7b: Re-run with COVID dummies (for comparison)
python -c "from src.models.regularized import run_regularized; run_regularized(with_covid_dummies=True)"

# Day 8: Dynamic Factor Model (3 vintages, ~5–10 min)
python -m src.models.dfm

# Day 12: Method-family combination forecast
python -m src.models.combination

# Day 9: Master evaluation (DM tests, COVID analysis)
python -m src.evaluation.master

# Day 10: Generate all four figures
python -m src.evaluation.figures
```

### View results

```bash
# Headline metrics table
cat results/tables/master_summary.csv

# Full results with DM test p-values
cat results/tables/master_results.csv

# Figures (PNG + PDF)
ls results/figures/
```

Open `notebooks/99_final_figures.ipynb` in Jupyter to view and regenerate all figures interactively.

---

## Project Structure

```
.
├── data/
│   ├── raw/           # Cached FRED parquets (gitignored)
│   └── processed/     # Transformed monthly panel + quarterly GDP growth
├── results/
│   ├── figures/       # fig1–fig4 as PNG and PDF
│   ├── forecasts/     # One parquet per model × vintage (95 files)
│   └── tables/        # master_results.csv, master_summary.csv, model-specific CSVs
├── src/
│   ├── config.py      # Series IDs, publication lags, date constants, paths
│   ├── data/
│   │   ├── fetch.py           # FRED pull with caching and rate-limit retry
│   │   ├── transforms.py      # Tcode transforms, panel builders
│   │   └── ragged_edge.py     # Publication-lag masker, vintage_as_of helper
│   ├── models/
│   │   ├── benchmarks.py      # AR(p) and random-walk
│   │   ├── bridge.py          # Bridge equations
│   │   ├── midas.py           # U-MIDAS
│   │   ├── regularized.py     # Lasso, ElasticNet (with optional COVID dummies)
│   │   └── dfm.py             # Dynamic Factor Model
│   ├── evaluation/
│   │   ├── cv.py              # Expanding-window OOS loop
│   │   ├── tests.py           # Diebold-Mariano test
│   │   ├── master.py          # Master evaluation script
│   │   └── figures.py         # Figure generators
│   └── utils/
│       └── plotting.py        # Shared style, colour palette, save_fig
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 99_final_figures.ipynb
├── tests/
│   └── test_ragged_edge.py    # Unit tests for the masker (5 tests)
├── docs/
│   ├── report.md          # Full research report
│   ├── Final Dashboard.html  # Interactive HTML explainer with charts
│   └── EXPLAINER.pdf      # PDF export of plain-language explainer
├── requirements.txt
└── .env.example       # Template — copy to .env and add FRED_API_KEY
```

---

## Key Design Decisions

- **Ragged edge**: fixed per-series publication lags in `config.PUBLICATION_LAGS_DAYS`; lags approximate typical BLS/Census/Fed release calendars. Actual release dates vary slightly month to month.
- **Partial aggregation (bridge/MIDAS)**: training uses complete quarterly aggregates; forecasting uses the mean of available months. This is a deliberate simplification documented in `src/models/bridge.py`.
- **LOCF imputation (MIDAS, regularized)**: unreleased within-quarter monthly values are filled with the last observed value. This preserves the K=3 lag structure without discarding observations.
- **TimeSeriesSplit CV**: all hyperparameter selection uses `TimeSeriesSplit` with no shuffling. Plain KFold is never used.
- **DFM refit cadence**: EM algorithm runs every 4 quarters; Kalman smoother with cached parameters otherwise.
- **COVID handling**: results reported for full, pre-COVID, and ex-COVID samples. COVID dummies tested for regularized models only.
