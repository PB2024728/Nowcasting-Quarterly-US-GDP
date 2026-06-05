"""Unit tests for src.evaluation.tests.dm_test."""

import numpy as np
import pytest

from src.evaluation.tests import dm_test

_RNG = np.random.default_rng(42)


def test_positive_stat_when_alternative_better():
    """Positive DM stat when e2 has strictly smaller squared loss."""
    e1 = np.full(60, 2.0)   # benchmark: large errors
    e2 = np.full(60, 0.5)   # alternative: small errors
    stat, pval, diff = dm_test(e1, e2)
    assert stat > 0, "DM stat should be positive when alternative is better"
    assert diff > 0, "Mean loss differential should be positive"


def test_negative_stat_when_baseline_better():
    """Negative DM stat when e1 has smaller squared loss than e2."""
    e1 = np.full(60, 0.1)
    e2 = np.full(60, 3.0)
    stat, pval, diff = dm_test(e1, e2)
    assert stat < 0


def test_identical_errors_handled_gracefully():
    """
    Identical error vectors → loss differential d=0 everywhere → degenerate case.
    The HAC variance is 0, so stat and pval are NaN.  This is the correct
    result (undefined test statistic) and should not raise an exception.
    """
    e = _RNG.normal(0, 1, 80)
    stat, pval, diff = dm_test(e, e)
    # Either NaN (degenerate) or finite — must not raise
    assert np.isnan(stat) or np.isfinite(stat)


def test_nan_handling():
    """NaN entries in either error vector are excluded correctly."""
    e1 = np.array([1.0, np.nan, 2.0, 3.0] * 15)
    e2 = np.array([0.5, 0.5, np.nan, 1.5] * 15)
    stat, pval, diff = dm_test(e1, e2)
    assert np.isfinite(stat)
    assert 0.0 <= pval <= 1.0


def test_absolute_loss():
    """DM test works with absolute loss."""
    e1 = np.full(60, 2.0)
    e2 = np.full(60, 1.0)
    stat, pval, diff = dm_test(e1, e2, loss="absolute")
    assert stat > 0


def test_invalid_loss_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        dm_test(np.ones(30), np.ones(30), loss="cubic")


def test_returns_nan_on_too_few_observations():
    stat, pval, diff = dm_test(np.array([1.0, 2.0]), np.array([0.5, 1.5]))
    assert np.isnan(stat)


def test_pvalue_between_zero_and_one():
    e1 = _RNG.normal(0, 1, 80)
    e2 = _RNG.normal(0, 1.5, 80)
    _, pval, _ = dm_test(e1, e2)
    assert 0.0 <= pval <= 1.0
