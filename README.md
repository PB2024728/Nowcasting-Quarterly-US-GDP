# Nowcasting Quarterly U.S. GDP

This project investigates whether quarterly U.S. real GDP growth can be nowcast more accurately than simple time-series benchmarks using a set of monthly macroeconomic indicators sourced from FRED. Four method families are compared — bridge equations, MIDAS regression, regularized regression (Lasso / ElasticNet), and a dynamic factor model — against AR(p) and random-walk baselines. The evaluation period runs from 2005 Q1 to the present using an expanding-window out-of-sample scheme, and results are reported for three within-quarter data vintages (end of month 1, 2, and 3) to quantify how forecast accuracy improves as the quarter's data arrive. COVID quarters receive explicit treatment throughout.

## Reproduction

1. **Clone** the repository and `cd` into it.
2. **Create environment**: `py -m venv .venv` then activate (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Unix).
3. **Install**: `pip install -r requirements.txt`
4. **Set API key**: copy `.env.example` to `.env` and fill in `FRED_API_KEY` (free at <https://fred.stlouisfed.org/docs/api/api_key.html>).
5. **Run pipeline** in day order:
   - `python -m src.data.fetch` — download and cache FRED data
   - `python -m src.data.transforms` — apply stationarity transforms
   - `python -m src.models.benchmarks` — fit AR and RW baselines
   - `python -m src.models.bridge` — bridge equation nowcasts
   - `python -m src.models.midas` — MIDAS nowcasts
   - `python -m src.models.regularized` — Lasso / ElasticNet nowcasts
   - `python -m src.models.dfm` — dynamic factor model nowcasts
   - `python -m src.evaluation.master` — consolidated metrics and DM tests
   - `python -m src.evaluation.figures` — regenerate all figures
6. **View results**: open `notebooks/99_final_figures.ipynb` for the full narrative.

Headline results are in `results/tables/master_results.csv`.
