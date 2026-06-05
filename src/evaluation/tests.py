"""
Statistical tests for forecast evaluation.

dm_test(e1, e2)
    Diebold-Mariano test for equal predictive accuracy (Diebold & Mariano 1995).
    Uses HAC standard errors via statsmodels OLS with cov_type='HAC'.
    Bandwidth defaults to ceil(T^(1/3)) following the literature.

    H0: E[L(e1_t) - L(e2_t)] = 0  (equal loss)
    H1: E[L(e1_t) - L(e2_t)] != 0  (unequal loss)

    A positive DM statistic means e2 has lower average loss than e1
    (model 2 is more accurate than model 1).

    Typical usage: dm_test(e_benchmark, e_alternative)
    Positive stat + small p-value → alternative significantly beats benchmark.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm


def dm_test(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: str = "squared",
    maxlags: int | None = None,
) -> tuple[float, float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy with HAC standard errors.

    Parameters
    ----------
    e1 : forecast errors from model 1 (benchmark)
    e2 : forecast errors from model 2 (alternative)
    loss : 'squared' or 'absolute'
    maxlags : HAC bandwidth; defaults to ceil(T^(1/3))

    Returns
    -------
    (dm_stat, p_value, mean_loss_differential)
        dm_stat   > 0  →  alternative (e2) has lower loss
        p_value   < 0.05  →  reject equal predictive accuracy at 5%
        mean_loss_differential = mean(L(e1) - L(e2))
    """
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()

    if len(e1) != len(e2):
        raise ValueError(f"Length mismatch: {len(e1)} vs {len(e2)}")

    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss '{loss}'. Use 'squared' or 'absolute'.")

    # Align: keep only rows where both errors are finite
    valid = np.isfinite(d)
    d_clean = d[valid]
    T = len(d_clean)

    if T < 5:
        return np.nan, np.nan, np.nan

    if maxlags is None:
        maxlags = int(np.ceil(T ** (1.0 / 3.0)))

    # OLS of d on a constant with Newey-West (HAC) standard error
    X = np.ones((T, 1))
    res = sm.OLS(d_clean, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": True},
    )

    return float(res.tvalues[0]), float(res.pvalues[0]), float(res.params[0])
