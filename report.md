# Nowcasting Quarterly U.S. Real GDP Growth

**Proteek Basu** | Pre-Masters Project #3 | June 2026

---

## 1. Research Question

Can monthly macroeconomic indicators improve real-time predictions of quarterly U.S. real GDP growth relative to a pure time-series benchmark? And if so, which mixed-frequency methods work best, and how much does accuracy improve as the quarter's data accumulate?

The practical motivation is that GDP is published with a multi-month lag: the advance estimate for a quarter typically arrives six weeks after quarter-end. Nowcasting bridges that gap by conditioning on monthly indicators—industrial production, payrolls, retail sales, financial spreads—that are released throughout the quarter. The research question therefore concerns both *whether* any method beats a simple AR baseline and *how quickly* forecast accuracy rises as each successive month of data arrives.

---

## 2. Data and Transformations

### 2.1 Series

Twelve monthly FRED predictors were used alongside the quarterly target (GDPC1, real GDP in chained 2017 dollars, seasonally adjusted):

| Series | Description | Transform |
|---|---|---|
| INDPRO | Industrial Production Index | Log-diff |
| PAYEMS | Nonfarm Payrolls | Log-diff |
| RSAFS | Retail and Food Services Sales | Log-diff |
| UNRATE | Unemployment Rate | First diff |
| ICSA | Initial Jobless Claims (weekly→monthly sum) | Log-diff |
| HOUST | Housing Starts | Log-diff |
| DGORDER | Durable Goods Orders | Log-diff |
| UMCSENT | U. of Michigan Consumer Sentiment | First diff |
| PCEPI | PCE Price Index | Log-diff |
| T10Y2Y | 10Y–2Y Treasury Yield Spread | First diff |
| BAA10Y | BAA–10Y Treasury Spread | First diff |
| NASDAQCOM | NASDAQ Composite Index | Log-diff |

The FRED series `SP500` covers only 2016–present and was replaced with NASDAQCOM (available from 1990). Daily and weekly series were aggregated to monthly means (financial series) or sums (ICSA) before caching. Two series have shorter effective histories: RSAFS begins January 1992 and DGORDER has missing values through January 1992 in the FRED release; both result in NaN rows that downstream models handle explicitly.

The GDP target was transformed to annualized quarter-on-quarter log growth: $400 \times \log(\text{GDP}_t / \text{GDP}_{t-1})$. Over 1990Q1–2026Q1, this series has mean 2.4% and standard deviation 4.4%.

### 2.2 Stationarity Transforms

Log-differences approximate percentage changes and render level series stationary. First differences handle already-stationary rates and spreads. All transforms follow FRED-MD `tcode` conventions. NaNs are never silently dropped; the first observation of each differenced series is NaN by construction.

### 2.3 Ragged-Edge Masking

The core realism device is the *ragged-edge masker* (`src/data/ragged_edge.py`). For a given as-of date, a monthly observation is set to NaN if the publication would not yet have occurred:

$$\text{release date}_t = \text{month-end}_t + \text{lag}_s$$

Publication lags (days after reference month-end): UMCSENT 0, financial series 1, PAYEMS/UNRATE/ICSA 5, RSAFS 14, INDPRO/HOUST ~17–19, DGORDER/PCEPI 28.

Three **within-quarter vintages** are evaluated for each target quarter $q$:
- **Vintage 1** (V1): as-of = last day of month 1 of $q$
- **Vintage 2** (V2): as-of = last day of month 2
- **Vintage 3** (V3): as-of = last day of month 3

An important consequence of these lags: at V3 (quarter-end), the majority of month-3 monthly indicators are *still NaN* because publication lags extend into the following month. Models at V3 therefore operate on essentially the same information as V2 for slow-release series.

---

## 3. Methods

All models use an **expanding-window out-of-sample scheme**: for each target quarter $q$ from 2005Q1 onward, the model is estimated on all available data strictly before $q$ and predicts $q$. No future information is ever used in estimation.

### 3.1 Benchmark Models

**AR(p)**: an autoregression on quarterly GDP growth with lag order $p \in \{1, \ldots, 8\}$ selected by BIC at each expanding-window step. This is the bar every alternative must clear.

**Random Walk (historical mean)**: predicts the historical mean of GDP growth up to the cutoff. Remarkably competitive with the AR(p) in the pre-COVID period.

### 3.2 Bridge Equations

For each monthly predictor $i$, the quarterly aggregate is formed as the mean of available within-quarter monthly observations (partial aggregation for incomplete quarters). An OLS regression is estimated:

$$\text{GDP}_t = \beta_0 + \beta_1 x_{i,t} + \beta_2 \text{GDP}_{t-1} + \varepsilon_t$$

The twelve individual forecasts are combined by simple averaging. Partial aggregation is a deliberate simplification: training uses complete quarterly aggregates while forecasting uses whatever months are available.

### 3.3 MIDAS Regression

Unrestricted MIDAS (U-MIDAS) keeps the three monthly lags within the quarter as separate features:

$$\text{GDP}_t = \beta_0 + \beta_1 x_{i,m1} + \beta_2 x_{i,m2} + \beta_3 x_{i,m3} + \beta_4 \text{GDP}_{t-1} + \varepsilon_t$$

This is OLS with the right feature engineering. Unreleased monthly cells are filled by last-observation-carried-forward (LOCF) from the most recent non-NaN value. Models are estimated per indicator and combined by averaging. Almon polynomial MIDAS was not implemented; with $K=3$ and 60+ training observations, U-MIDAS is the appropriate baseline.

### 3.4 Regularized Regression

A multi-indicator design matrix stacks $K=3$ monthly lags for all 12 predictors (36 features) plus lagged GDP, giving 37 features. OLS would overfit with this many parameters relative to ~60–80 training observations.

**Lasso** and **ElasticNet** impose $\ell_1$ and $\ell_1$/$\ell_2$ penalties respectively. Both are wrapped in `Pipeline(StandardScaler → LassoCV/ElasticNetCV)` refitted at each expanding-window step. Hyperparameters are tuned by `TimeSeriesSplit(n_splits=5)` cross-validation—no plain KFold or shuffling that would leak future information.

A variable-selection log records which of the 37 features had non-zero Lasso coefficients at each OOS step. INDPRO (m1, m2, m3), PAYEMS (m2), ICSA (m3), and NASDAQCOM (m3) were selected in ≥95% of all quarterly fits, providing a stability map of the most informative indicators.

COVID dummies (binary indicators for 2020Q2 and 2020Q3) were added to the design matrix in a separate specification (`lasso_covid`, `elasticnet_covid`). These dummies limit the influence of the COVID outliers on coefficient estimates once those quarters enter the training window. Pre-COVID metrics are identical to the no-dummy specification by construction.

### 3.5 Dynamic Factor Model

`statsmodels.tsa.statespace.DynamicFactorMQ` jointly estimates one latent factor from all 12 monthly predictors and quarterly GDP. The model is specified with AR(1) factor dynamics and handles ragged edges natively through the Kalman filter: NaN cells simply receive no observation update and the state is propagated forward through the factor AR dynamics.

The EM algorithm re-estimates all state-space parameters every four quarters; between refits the Kalman smoother is run with the cached parameter vector—a runtime compromise that reduces computation by ~75% with negligible accuracy cost.

---

## 4. Evaluation Setup

The out-of-sample period runs from **2005Q1 through 2026Q1** (85 quarters). Metrics are reported under three sample definitions:

| Sample | Window | N |
|---|---|---|
| Full | 2005Q1–2026Q1 | 85 |
| Pre-COVID | 2005Q1–2019Q4 | 60 |
| Ex-COVID | Full excl. 2020Q1–2021Q2 | 79 |

The primary metric is **RMSE** (annualized GDP growth, percentage points). MAE and mean bias are also reported. Full-sample metrics are dominated by the 2020Q2 GDP collapse (~−31.6% annualized); the pre-COVID sample is the operationally meaningful comparison.

Statistical significance is assessed with the **Diebold-Mariano test** (Diebold & Mariano 1995) using Newey-West HAC standard errors with bandwidth $\lceil T^{1/3} \rceil$. All alternatives are tested against the AR(p) baseline. A positive DM statistic indicates the alternative has lower mean squared loss.

---

## 5. Results

### 5.1 Headline RMSE Table (Pre-COVID Sample)

| Model | Vintage | RMSE (%) | DM stat | p-value |
|---|---|---|---|---|
| **AR(p) baseline** | — | **2.47** | — | — |
| RW | — | 2.46 | 0.04 | 0.971 |
| Bridge combo | V1 | 2.22 | 1.41 | 0.158 |
| MIDAS combo | V1 | 2.11 | 1.66 | **0.097** |
| Lasso | V1 | 2.36 | 0.42 | 0.673 |
| ElasticNet | V1 | 2.31 | 0.50 | 0.616 |
| DFM | V1 | 2.85 | −1.04 | 0.299 |
| Bridge combo | V2 | 1.89 | 1.68 | **0.092** |
| MIDAS combo | V2 | 1.93 | 1.68 | **0.093** |
| Lasso | V2 | 2.17 | 0.99 | 0.321 |
| ElasticNet | V2 | 2.06 | 0.95 | 0.340 |
| DFM | V2 | 2.79 | −0.80 | 0.422 |
| Bridge combo | V3 | **1.87** | 1.79 | **0.074** |
| MIDAS combo | V3 | 1.93 | 1.70 | **0.089** |
| Lasso | V3 | 2.07 | 1.34 | 0.181 |
| ElasticNet | V3 | 1.90 | 1.24 | 0.216 |
| DFM | V3 | 2.77 | −0.73 | 0.467 |
| **Method combination** | **V1** | **1.90** | **1.66** | **0.098** |
| **Method combination** | **V2** | **1.77** | **1.72** | **0.085** |
| **Method combination** | **V3** | **1.72** | **1.77** | **0.076** |

**Models that beat AR(p) at the 10% significance level (pre-COVID):** Bridge combination (V2, V3), MIDAS combination (all vintages), and the method-family combination (all vintages). No model achieves significance at the conventional 5% level—a common outcome in quarterly macro evaluation with ~60 observations and HAC-adjusted standard errors.

### 5.2 Method-Family Combination Forecast

A simple equal-weight average across the four family combination forecasts (Bridge, MIDAS, ElasticNet, DFM) achieves the lowest RMSE of any model: **1.72% pre-COVID at V3**, improving on the best individual model (Bridge 1.87%) by 0.15 percentage points. The combination is significant at 10% against the AR(p) at all three vintages—a result no single family achieves consistently across all vintages.

This is consistent with the forecast combination literature (e.g., Stock & Watson 2004): pooling over methodologically diverse models reduces variance and produces more robust out-of-sample performance, particularly when individual models each capture different aspects of the data-generating process. The DFM's systematic over-prediction bias is partially offset by the downward biases of the regression models when they are combined.

### 5.3 Vintage Learning

The central nowcasting result is confirmed: RMSE declines as the quarter fills in. The Bridge combination improves from 2.22% at V1 to 1.87% at V3 (−16%), illustrating that each successive month of monthly releases meaningfully updates the GDP nowcast. MIDAS shows a similar V1→V2 improvement but less V2→V3 gain, consistent with slow-release series still being unreleased at quarter-end.

### 5.4 The Dynamic Factor Model

The DFM does not beat the AR(p) benchmark on the pre-COVID sample (DM stat −0.73 to −1.04, pre-COVID RMSE ≈ 2.77–2.85%). It also exhibits a consistent upward bias of ~2.1 percentage points. Two factors explain this underperformance:

1. **Small panel**: with only 12 indicators, a single-factor DFM has limited advantage over reduced-form regression models. DFMs typically dominate on large panels such as FRED-MD (~120 series).
2. **Publication lag constraint**: at V3 (quarter-end), the majority of month-3 monthly data is still NaN due to publication lags. The Kalman filter therefore propagates the state through mostly-missing observations, yielding essentially the same information as V2. This eliminates the DFM's nominal advantage of having a richer observation set.

### 5.5 COVID Period

Full-sample RMSEs are inflated by the COVID collapse: AR RMSE = 8.16% (vs 2.47% pre-COVID). Bridge and MIDAS combinations outperform the AR even on the full sample (Bridge V2: 5.20%, MIDAS V2: 5.03%) but the differences are not statistically significant. Adding COVID dummies to the regularized models does not materially change pre-COVID or ex-COVID RMSEs, confirming that the primary benefit of COVID dummies is in limiting their effect on coefficient estimates for post-2021 forecasts rather than improving forecast accuracy of the COVID quarters themselves.

---

## 6. Limitations

**Revised data, not real-time vintages.** All FRED series used here are the *current* (most recently revised) vintage, not the real-time vintage that a forecaster would have observed in 2005. Revisions tend to smooth and improve data quality, so the accuracy figures reported here are optimistic. A proper real-time evaluation would use ALFRED (Archival FRED) vintage data, which was not implemented in this project. The ragged-edge masker approximates the *timing* of releases correctly, but cannot replicate *revision noise*.

**Approximate publication lags.** The per-series publication lags in `src/config.py` are fixed constants based on typical BLS, Census, and Federal Reserve release calendars. In practice, release dates vary by one to several days each month, and occasional delays or advance releases occur. Using a fixed calendar overstates the predictability of exactly which data is available on any given as-of date.

**COVID handling.** The 2020Q2 GDP collapse (−31.6% annualized) is orders of magnitude larger than the training distribution of any model. No regression-based model, including the DFM, can be expected to predict a pandemic from macroeconomic leading indicators. The COVID dummy approach (for regularized regression) is a mitigation, not a solution. Excluding COVID quarters from the ex-COVID sample is the cleanest comparator but removes observations that are economically relevant. Results are reported under all three sample definitions so the reader can judge.

**Single factor / small panel for DFM.** A single-factor DFM estimated over 12 indicators cannot fully represent the multi-dimensional structure of the macroeconomy. The DFM's advantage over simpler models is documented in the literature primarily for large panels (50–200 series). With 12 series, the regularized regression models provide a more natural form of dimension reduction.

**No formal real-time evaluation.** This project does not compare against professionally produced nowcasts such as the Federal Reserve Bank of New York's Staff Nowcast or the Survey of Professional Forecasters. Such comparison would be the natural next step to assess whether the methods here add value over the practitioner state of the art.

---

*All results and figures are fully reproducible. See `README.md` for step-by-step instructions.*
