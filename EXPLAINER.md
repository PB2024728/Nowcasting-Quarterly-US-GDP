# Nowcasting U.S. GDP — Plain Language Explainer

**What this project does, how it works, and what we found**

---

## The Big Idea

Every three months, the U.S. government publishes an official estimate of how fast the economy grew. This number — GDP growth — is one of the most watched economic statistics in the world. The problem is that it arrives about **six weeks after the quarter ends**, and the first estimate is still subject to revision for years afterward.

**The hypothesis:** Monthly economic indicators — things like factory output, job numbers, and stock prices — get published throughout the quarter as it happens. Can we use these early signals to predict the GDP number *before* the official estimate is released? And if so, how much more accurate are these predictions compared to simply extrapolating past GDP patterns?

This project tests that hypothesis systematically. We built and compared five different forecasting approaches, evaluated them rigorously on 20 years of data, and quantified which methods work, by how much, and at what level of statistical confidence.

---

## Why This Is Hard

A few things make GDP nowcasting genuinely challenging:

**1. The timing mismatch.** GDP is quarterly. Monthly indicators arrive throughout the quarter. A GDP forecast made in early January uses different data than one made in late March, even though both are predicting the same Q1 number. We need to handle this "mixed frequency" problem explicitly.

**2. The publication delay.** Data doesn't arrive the moment the reference month ends. Payroll numbers come out about five days after month-end. Factory output takes about seventeen days. Durable goods orders take nearly a month. We need to simulate, for any given date, exactly which data would actually have been available to a real forecaster — and which would still be missing.

**3. The rarity of recessions.** We only have about 60 "normal" (pre-COVID) quarters to evaluate on. That's not a lot of data points, which means statistical tests have limited power — it's hard to definitively prove one method beats another.

**4. COVID.** The 2020Q2 GDP collapse (−31.6% annualized) is so extreme that it distorts every metric. No model trained on peacetime data could have predicted a pandemic. We handle this by reporting results three ways: including COVID, excluding COVID entirely, and for the pre-COVID period only.

---

## The Data

We used 13 series from the Federal Reserve's FRED database, all freely available:

**The target (what we're predicting):**
- `GDPC1` — Real GDP, quarterly, inflation-adjusted

**The 12 monthly predictors (our leading indicators):**

| Series | What it measures |
|---|---|
| INDPRO | Industrial production — how much stuff factories are making |
| PAYEMS | Nonfarm payrolls — total number of people on payrolls |
| RSAFS | Retail sales — how much consumers are spending |
| UNRATE | Unemployment rate |
| ICSA | Initial jobless claims — how many people filed for unemployment this week |
| HOUST | Housing starts — new homes beginning construction |
| DGORDER | Durable goods orders — orders for long-lasting manufactured goods |
| UMCSENT | Consumer sentiment — how optimistic people feel about the economy |
| PCEPI | PCE inflation — the Fed's preferred price index |
| T10Y2Y | Yield curve spread — difference between 10-year and 2-year Treasury rates |
| BAA10Y | Credit spread — extra interest rate corporations pay vs. the government |
| NASDAQCOM | NASDAQ stock index (used as a proxy for equity market conditions) |

Before feeding these into any model, each series was transformed to be stationary (no trend). Level series like industrial production were log-differenced (approximate percentage changes). Rate series like unemployment were first-differenced. GDP growth became `400 × log(GDP_t / GDP_{t-1})` — annualized quarterly percent growth.

---

## Day-by-Day: What We Built

### Day 1 — Project Scaffolding
Set up the code structure, installed all dependencies, created a configuration file that stores all the settings in one place (series IDs, publication lags, date ranges, file paths), and initialized the git repository. Nothing fancy — just making sure every subsequent day has a clean, organized place to put things.

### Day 2 — Getting the Data
Built a data fetcher that downloads all 13 series from FRED and saves them to disk. The key feature: once downloaded, re-running the code reads from disk (fast, ~1 second) rather than hitting the internet again. Daily series (stock prices, yield spreads) were averaged down to monthly. Weekly claims data was summed to monthly. One discovery: FRED's `SP500` series only goes back to 2016 — too short for our 1990 sample. We replaced it with `NASDAQCOM`, which covers the full period.

### Day 3 — The Ragged Edge
This is the most important piece of infrastructure in the project. The "ragged edge" is the name for what a real forecaster actually sees: a table of data where the most recent months are partially filled in, because not everything has been published yet.

We built a masker that takes any "as-of" date and sets cells to missing (NaN) if the underlying data release wouldn't have happened yet on that date. For example: if you're forecasting as of January 31st, industrial production for January is still missing (it won't be published until mid-February), but December's number is available.

We also defined three **vintages** per quarter:
- **Vintage 1 (V1):** as of the last day of month 1 of the quarter — earliest possible forecast
- **Vintage 2 (V2):** as of the last day of month 2 — middle of the quarter
- **Vintage 3 (V3):** as of the last day of month 3 — just as the quarter ends

Five unit tests were written to verify the masker works correctly.

### Day 4 — The Benchmarks
Built the two models that everything else has to beat:

**AR(p) model:** Fits an autoregression on quarterly GDP growth — uses only past GDP to predict future GDP. The lag length (how many past quarters to include) is chosen automatically using an information criterion (BIC) at each step. This is the primary benchmark.

**Random walk:** Simply predicts the historical average of GDP growth. Surprisingly competitive.

The evaluator runs an **expanding-window** backtest: start in 2005, use all data up to quarter $t-1$ to predict quarter $t$, then add $t$ to the training set and repeat. This faithfully mimics what a real forecaster would have done at each point in time.

**The bar to beat:** AR(p) pre-COVID RMSE = **2.47%**

*(RMSE is "root mean squared error" — roughly, the typical size of the forecast mistake, in percentage points of annualized GDP growth.)*

### Day 5 — Bridge Equations
The simplest mixed-frequency approach. For each monthly indicator separately:
1. Aggregate the monthly data to quarterly frequency (just take the average of available months)
2. Regress quarterly GDP growth on the indicator plus last quarter's GDP growth
3. Use the fitted equation to predict the current quarter

The key insight here is the **partial aggregation** rule: if only one month of the quarter is available, use that one month's value as the "quarterly aggregate." It's an approximation, but it works reasonably well.

We built 12 individual bridge equations (one per indicator), then combined them by simple averaging. This "combination forecast" consistently outperformed any individual bridge equation.

**Result:** Bridge combination V3 pre-COVID RMSE = **1.87%** — better than AR by 24%.

### Day 6 — MIDAS Regression
"MIDAS" stands for Mixed Data Sampling. Instead of collapsing monthly data into a quarterly average, it uses the three individual monthly observations as separate features in the regression:

```
GDP_growth = b0 + b1×(indicator_month1) + b2×(indicator_month2) + b3×(indicator_month3) + b4×(GDP_last_quarter)
```

This is "unrestricted MIDAS" (U-MIDAS) — OLS with smarter feature engineering. It lets the model weight the three months differently, which can be valuable: the third month of a quarter often carries more signal than the first.

For the vintage problem: if a month's data isn't released yet, we fill it with the most recent available value (carry the last observation forward). This placeholder keeps the three-feature structure intact.

**Result:** MIDAS combination V1 pre-COVID RMSE = **2.11%** — beats bridge at the earliest vintage, because the granular monthly structure matters more when there's less data.

### Day 7 — Regularized Regression (Lasso and ElasticNet)
Here we get ambitious: instead of one indicator at a time, we throw all 12 indicators into a single model simultaneously, using 3 monthly lags each = **37 features** (plus an intercept).

With only 60–80 training observations and 37 features, ordinary regression would overfit badly. **Lasso** solves this by forcing most coefficients to exactly zero — it automatically selects which indicators matter and ignores the rest. **ElasticNet** is a variant that handles correlated predictors more gracefully.

Key technical requirements:
- Features must be **standardized** before fitting (so that coefficients are comparable)
- The regularization strength (how aggressively to shrink coefficients) must be chosen by cross-validation — but the cross-validation must respect time order (no peeking at the future)
- All of this is wrapped in a sklearn `Pipeline` that refits from scratch at each expanding-window step

We also logged which variables Lasso selected at each step. Four variables were selected in ≥95% of all OOS quarters: `INDPRO_m1`, `PAYEMS_m2`, `ICSA_m3`, and `NASDAQCOM_m3` — the first month's industrial production, the second month's payrolls, the third month's claims, and the third month's stock market. Economically intuitive.

**Result:** ElasticNet V3 pre-COVID RMSE = **1.90%** — competitive with bridge.

### Day 8 — Dynamic Factor Model (DFM)
The most sophisticated model. The idea: instead of treating each indicator as a separate predictor, there is some underlying hidden "economic activity" factor that drives all of them simultaneously. Industrial production goes up when the economy is strong. Claims go down. The stock market rises. All because of the same underlying force.

The DFM estimates this hidden factor using the **Kalman filter** — an algorithm originally developed for rocket guidance systems that recursively updates a state estimate as new observations arrive. Crucially, it handles missing data natively: when a cell is NaN (data not yet released), it simply skips the update step for that indicator rather than requiring any imputation.

We used statsmodels' `DynamicFactorMQ`, which is purpose-built for mixed-frequency (monthly + quarterly) nowcasting. We specified one factor with AR(1) dynamics and refitted the model parameters every four quarters to save computation time.

**Result:** DFM V3 pre-COVID RMSE = **2.77%** — *worse* than the AR benchmark. This is an important finding and is discussed below.

### Day 9 — Rigorous Evaluation
Pulled everything together with formal statistical testing. The **Diebold-Mariano test** asks: is the difference in accuracy between two models statistically significant, or just noise? It uses HAC (heteroskedasticity and autocorrelation consistent) standard errors, which are appropriate for time-series data.

Results were reported under three sample definitions: full OOS (2005–present), pre-COVID (2005–2019Q4), and ex-COVID (excluding 2020Q1–2021Q2). We also tested whether adding COVID quarter dummies to the Lasso/ElasticNet models improved performance.

### Day 10 — Figures
Generated four publication-quality figures:
1. **RMSE grouped bar chart** — how each model performs across the three vintages
2. **Time-series plot** — realized GDP growth vs. key model nowcasts, 2005–present
3. **RMSE vs. vintage** — how accuracy improves as the quarter fills in
4. **Lasso selection map** — which variables were persistently selected

### Day 11 — Report
Wrote the full research report (`report.md`) documenting the methodology, results, and limitations. Updated this README with reproduction instructions.

### Day 12 — Combination Forecast and Final Polish
**Stretch goal:** Instead of choosing a single best model, what if we averaged across all four method families? This is the "forecast combination" — a technique that's been shown to be robustly good in the forecasting literature, because different models make different kinds of mistakes, and averaging tends to cancel them out.

We averaged the Bridge combination, MIDAS combination, ElasticNet, and DFM forecasts. Expanded the test suite from 5 to 19 tests. Tagged version `v1.0`.

---

## The Models Explained Simply

Here's an intuitive comparison of the five approaches:

| Approach | What it does | Analogy |
|---|---|---|
| **AR(p)** | Uses only past GDP to predict future GDP | Predicting tomorrow's weather from only today's temperature |
| **Bridge equations** | Each economic indicator gets its own simple regression; combine the 12 predictions | 12 experts give opinions; take the average |
| **MIDAS** | Like bridge, but keeps monthly observations separate instead of averaging them | Same 12 experts, but they can break their opinion into three parts (early, middle, late month) |
| **Lasso/ElasticNet** | All 12 indicators in one model; computer automatically zeroes out the unimportant ones | One expert reads all 12 indicators but ignores the noise |
| **DFM** | Assumes a hidden "economic health" factor drives everything; the Kalman filter estimates it | A doctor diagnoses overall health from 12 vital signs simultaneously |
| **Combination** | Average the four family predictions | Committee vote — no single expert dominates |

---

## The Results

### The headline numbers (pre-COVID, 2005Q1–2019Q4)

| Model | Month 1 nowcast | Month 2 nowcast | Month 3 nowcast |
|---|---|---|---|
| AR(p) — the bar to beat | 2.47% | 2.47% | 2.47% |
| Bridge combination | 2.22% | 1.89% ✓ | 1.87% ✓ |
| MIDAS combination | 2.11% ✓ | 1.93% ✓ | 1.93% ✓ |
| Lasso | 2.36% | 2.17% | 2.07% |
| ElasticNet | 2.31% | 2.06% | 1.90% |
| DFM | 2.85% | 2.79% | 2.77% |
| **Method combination** | **1.90%** ✓ | **1.77%** ✓ | **1.72%** ✓ |

*Numbers are RMSE in annualized percent. Lower is better. ✓ = statistically significantly better than AR at the 10% level.*

### What the numbers mean

A typical pre-COVID GDP quarter grew at around 2–3%. The AR model's forecast error was ±2.47 percentage points. The best model (the combination at vintage 3) brought that down to ±1.72 percentage points — a **30% reduction in forecast error**.

To put that concretely: if GDP is actually going to grow at 2.5%, the AR model's forecast might be anywhere from 0% to 5%. The combination model narrows that to roughly 0.8% to 4.2%.

### The "vintage learning" story

The central result of nowcasting is confirmed: forecasts improve as the quarter fills in. Looking at the Bridge combination:

- **End of Month 1:** RMSE = 2.22% (barely better than the AR baseline)
- **End of Month 2:** RMSE = 1.89% (much better — payrolls and industrial production are now in)
- **End of Month 3:** RMSE = 1.87% (marginally better still)

Most of the information gain happens between Month 1 and Month 2. The improvement from Month 2 to Month 3 is smaller than you'd expect, because most monthly series aren't actually published by the last day of the third month — they only come out days to weeks later. So "end of month 3" is informationally closer to "end of month 2" than the labels suggest.

### Why the DFM underperforms

The DFM was expected to be the "flagship" model — it's the workhorse of central bank nowcasting teams. But here it's *worse* than the simple AR benchmark (2.77% vs 2.47%). Why?

Two reasons:

1. **Too few indicators.** Central banks typically run DFMs with 50–200 indicators. With only 12, there's not enough signal for the latent factor structure to add value over simple regression.

2. **The publication lag trap.** At the end of the quarter (Month 3), most monthly data is *still missing* because publication lags extend into the following month. The DFM's Kalman filter propagates the state through these NaN months using only the factor dynamics — which is essentially the same as the Month 2 information set. The DFM has no advantage over simpler models that also use Month 1–2 data.

The DFM also shows a systematic upward bias (~2.1 percentage points pre-COVID), suggesting the factor structure is over-weighting signals of economic strength in the data.

### Why the combination wins

The method combination is the best overall result (1.72% pre-COVID RMSE). This happens because:

- The regression models (Bridge, MIDAS) consistently *under-predict* GDP growth (negative bias)
- The DFM consistently *over-predicts* (positive bias)
- When you average them, the biases partially cancel each other out
- The average also benefits from the combination reducing prediction variance — "diversification" of forecast errors

The combination is significantly better than the AR benchmark at the 10% level for all three vintages. No individual method achieves this consistency.

### What didn't work

**Nothing is significant at the 5% level** — the gold standard for statistical significance. With only 60 quarterly observations in the pre-COVID sample, the confidence intervals are wide. Even a genuinely superior model might not clear the 5% bar. This is a fundamental constraint of quarterly macroeconomic evaluation.

**COVID dummies didn't help much.** Adding explicit dummy variables for 2020Q2 and 2020Q3 to the regularized models made no difference to pre-COVID accuracy (by construction — those quarters weren't in the training set yet) and only modest differences elsewhere.

---

## Key Limitations

**We used revised data, not real-time data.** The FRED data we downloaded is the *current* (most recently revised) vintage. In 2005, a forecaster would have been working with noisier, less revised data. This means our accuracy numbers are probably slightly optimistic — the "ragged edge" problem of data revisions is not captured here, only the timing problem.

**Our publication lags are approximations.** We hardcoded fixed publication delays (e.g., "payrolls come out 5 days after month-end"). In reality, release dates shift by a few days each month. A proper real-time evaluation would use the ALFRED database to get exact historical release dates.

**We never compared against professional forecasters.** The Federal Reserve Bank of New York publishes weekly nowcasts. The Survey of Professional Forecasters publishes quarterly consensus estimates. We don't know whether our methods add anything beyond what professionals already do.

**The DFM used only 12 indicators.** A production DFM at a central bank would use 100+ series. Our results for the DFM reflect this small-panel limitation, not the approach's fundamental potential.

---

## The Big Takeaway

**Monthly economic data does help predict GDP — but modestly, and mainly in the middle of the quarter when the most informative series (payrolls, industrial production) have been released.**

The best result — a 30% reduction in forecast error relative to a pure time-series baseline — is economically meaningful but statistically fragile with only 60 quarterly observations. Bridge equations and MIDAS, despite being conceptually simple, perform as well as or better than the more sophisticated regularized regression and factor models on this dataset size. The combination of methods adds genuine value through bias diversification.

If you had to choose a single model for live nowcasting with 12 indicators, the method combination is the answer.

---

*Full technical details: see [`report.md`](report.md). Reproduction instructions: see [`README.md`](README.md).*
