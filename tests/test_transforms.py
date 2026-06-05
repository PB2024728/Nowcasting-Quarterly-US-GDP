"""Unit tests for src.data.transforms.apply_tcode."""

import numpy as np
import pandas as pd
import pytest

from src.data.transforms import apply_tcode


def _series(values):
    return pd.Series(values, dtype=float)


def test_tcode1_first_difference():
    s = _series([10.0, 12.0, 9.0, 15.0])
    result = apply_tcode(s, 1)
    assert np.isnan(result.iloc[0])                        # first obs lost to diff
    np.testing.assert_allclose(result.iloc[1:].values, [2.0, -3.0, 6.0])


def test_tcode5_log_difference():
    # log-diff of [1, e, e^2] should be [NaN, 1.0, 1.0]
    s = _series([1.0, np.e, np.e ** 2])
    result = apply_tcode(s, 5)
    assert np.isnan(result.iloc[0])
    np.testing.assert_allclose(result.iloc[1:].values, [1.0, 1.0], rtol=1e-10)


def test_tcode5_percent_change_approximation():
    # For small changes, log-diff ≈ pct change
    s = _series([100.0, 101.0, 102.0])
    result = apply_tcode(s, 5)
    assert np.isnan(result.iloc[0])
    assert abs(result.iloc[1] - 0.01) < 0.0002    # ≈ 1% change


def test_tcode1_preserves_length():
    s = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = apply_tcode(s, 1)
    assert len(result) == len(s)


def test_tcode5_preserves_length():
    s = _series([1.0, 2.0, 4.0, 8.0])
    result = apply_tcode(s, 5)
    assert len(result) == len(s)


def test_unsupported_tcode_raises():
    with pytest.raises(ValueError, match="Unsupported tcode"):
        apply_tcode(_series([1.0, 2.0]), tcode=2)
