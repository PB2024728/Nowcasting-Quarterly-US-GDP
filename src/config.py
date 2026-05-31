"""
Central configuration for the GDP nowcasting project.
All paths, series IDs, and date constants live here so every module
imports from one place rather than hard-coding strings.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
FORECASTS_DIR = RESULTS_DIR / "forecasts"

# ---------------------------------------------------------------------------
# FRED series
# ---------------------------------------------------------------------------

# Target variable
GDP_SERIES = "GDPC1"  # Real GDP, quarterly, seasonally adjusted

# Monthly predictor series and their FRED tcode-style transform codes.
# tcode 1 = first difference, tcode 5 = log first difference (approx % change)
PREDICTOR_SERIES = {
    "INDPRO":   {"desc": "Industrial Production Index",         "tcode": 5, "freq": "M"},
    "PAYEMS":   {"desc": "Nonfarm Payrolls",                    "tcode": 5, "freq": "M"},
    "RSAFS":    {"desc": "Retail and Food Services Sales",      "tcode": 5, "freq": "M"},
    "UNRATE":   {"desc": "Unemployment Rate",                   "tcode": 1, "freq": "M"},
    "ICSA":     {"desc": "Initial Jobless Claims (weekly)",     "tcode": 5, "freq": "W"},
    "HOUST":    {"desc": "Housing Starts",                      "tcode": 5, "freq": "M"},
    "DGORDER":  {"desc": "Durable Goods Orders",                "tcode": 5, "freq": "M"},
    "UMCSENT":  {"desc": "U of Michigan Consumer Sentiment",    "tcode": 1, "freq": "M"},
    "PCEPI":    {"desc": "PCE Price Index",                     "tcode": 5, "freq": "M"},
    "T10Y2Y":   {"desc": "10Y-2Y Treasury Yield Spread",        "tcode": 1, "freq": "D"},
    "BAA10Y":   {"desc": "BAA Corporate–10Y Treasury Spread",   "tcode": 1, "freq": "D"},
    # FRED's SP500 series only starts 2016; NASDAQCOM is the best-coverage daily equity proxy
    "NASDAQCOM": {"desc": "NASDAQ Composite Index",                  "tcode": 5, "freq": "D"},
}

ALL_SERIES = list(PREDICTOR_SERIES.keys()) + [GDP_SERIES]

# ---------------------------------------------------------------------------
# Publication lag dictionary (in days after the reference month ends).
# Used by the ragged-edge masker to determine which cells are visible
# on a given as-of date.
# Approximate values based on typical BLS / Census / Fed release calendars.
# ---------------------------------------------------------------------------
PUBLICATION_LAGS_DAYS = {
    "PAYEMS":   5,    # Employment situation, ~first Friday after reference month
    "ICSA":     5,    # Weekly claims, ~5 days after reference week
    "INDPRO":   17,   # Industrial production, ~mid-month+2
    "RSAFS":    14,   # Advance retail sales, ~2 weeks after month end
    "HOUST":    19,   # Housing starts, ~3 weeks after month end
    "DGORDER":  28,   # Durable goods orders, ~4 weeks after month end
    "UMCSENT":  0,    # Final sentiment released by end of reference month
    "PCEPI":    28,   # PCE / personal income, ~4 weeks after month end
    "UNRATE":   5,    # Employment situation same release as PAYEMS
    "T10Y2Y":   1,    # Financial series: essentially real-time (1-day lag)
    "BAA10Y":   1,
    "NASDAQCOM": 1,
}

# ---------------------------------------------------------------------------
# Date constants
# ---------------------------------------------------------------------------
SAMPLE_START = "1990-01-01"       # First observation to download
TRAIN_START = "1990Q1"            # First quarter in the training set
OOS_START = "2005Q1"              # First quarter in the out-of-sample evaluation
COVID_START = "2020Q1"            # First COVID-affected quarter
COVID_END = "2021Q2"              # Last quarter excluded in the COVID-clean sample

# ---------------------------------------------------------------------------
# Model hyperparameters (defaults — overridden by CV where applicable)
# ---------------------------------------------------------------------------
AR_MAX_LAGS = 8          # Maximum lag order considered when selecting AR(p) by BIC
MIDAS_K = 3              # Number of within-quarter monthly lags for (U-)MIDAS
DFM_N_FACTORS = 1        # Number of common factors in the dynamic factor model
DFM_REFIT_EVERY = 4      # Re-estimate DFM state-space matrices every N quarters
LASSO_CV_SPLITS = 5      # TimeSeriesSplit folds for Lasso / ElasticNet tuning
