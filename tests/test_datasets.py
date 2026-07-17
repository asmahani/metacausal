"""Tests for built-in datasets."""

from __future__ import annotations

import numpy as np
import pytest

from metacausal import CausalEnsemble
from metacausal.datasets import load_lalonde


class TestLalondeContinuous:
    def test_default_returns_continuous_y(self):
        X, T, Y = load_lalonde()
        assert Y.dtype == float
        # Wide range, continuous: many distinct values, not all in {0,1}.
        assert len(np.unique(Y)) > 10
        assert Y.min() >= 0  # earnings are non-negative

    def test_shape(self):
        X, T, Y = load_lalonde()
        assert X.shape[0] == T.shape[0] == Y.shape[0] == 445
        assert X.shape[1] == 10

    def test_treatment_is_binary(self):
        _, T, _ = load_lalonde()
        assert set(np.unique(T).tolist()) == {0, 1}

    def test_raw_returns_dataframe(self):
        df = load_lalonde(raw=True)
        assert hasattr(df, "columns")
        assert "treat" in df.columns
        assert "re78" in df.columns


class TestLalondeBinarize:
    def test_median_balanced_split(self):
        _, _, Y = load_lalonde(binarize_y="median")
        assert set(np.unique(Y).tolist()) == {0, 1}
        # Median split is approximately 50/50 (re78 has some ties at 0).
        frac_one = float(Y.mean())
        assert 0.45 < frac_one < 0.55

    def test_positive_threshold(self):
        _, _, Y = load_lalonde(binarize_y="positive")
        assert set(np.unique(Y).tolist()) == {0, 1}
        # ≈ 69% of LaLonde subjects have positive 1978 earnings.
        frac_one = float(Y.mean())
        assert 0.65 < frac_one < 0.75

    def test_dtype_is_integer_when_binarized(self):
        _, _, Y = load_lalonde(binarize_y="median")
        assert np.issubdtype(Y.dtype, np.integer)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="binarize_y"):
            load_lalonde(binarize_y="quartile")

    def test_raw_ignores_binarize_y(self):
        # raw=True returns the underlying DataFrame regardless of binarize_y.
        df1 = load_lalonde(raw=True)
        df2 = load_lalonde(raw=True, binarize_y="median")
        assert df1.equals(df2)


@pytest.mark.slow
def test_binary_default_ensemble_fits_on_binarized_lalonde():
    """Sanity check: every binary-pool component fits on real data and
    produces a finite ATE in the [-1, 1] risk-difference range."""
    X, T, Y = load_lalonde(binarize_y="median")

    ens = CausalEnsemble()  # outcome_type='auto' detects binary
    ens.fit(X, T, Y, random_state=42)

    result = ens.ate()
    assert ens._outcome_type == "binary"
    assert np.isfinite(result.ate)
    assert -1.0 <= result.ate <= 1.0

    # All seven components produced a finite, in-range ATE.
    assert len(result.component_estimates) == 7
    for name, est in result.component_estimates.items():
        assert np.isfinite(est.ate), f"{name} returned non-finite ATE"
        assert -1.0 <= est.ate <= 1.0, (
            f"{name} ATE={est.ate} outside the risk-difference range"
        )
