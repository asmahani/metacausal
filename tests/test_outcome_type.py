"""Tests for metacausal.outcome_type.infer_outcome_type."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metacausal import infer_outcome_type


class TestBinaryDetection:
    def test_int_zero_one(self):
        assert infer_outcome_type(np.array([0, 1, 0, 1])) == "binary"

    def test_float_zero_one(self):
        assert infer_outcome_type(np.array([0.0, 1.0, 0.0])) == "binary"

    def test_bool_dtype(self):
        assert infer_outcome_type(np.array([True, False, True])) == "binary"

    def test_python_list(self):
        assert infer_outcome_type([0, 1, 0, 1]) == "binary"

    def test_pandas_series_int(self):
        assert infer_outcome_type(pd.Series([0, 1, 0])) == "binary"

    def test_pandas_series_bool(self):
        assert infer_outcome_type(pd.Series([True, False, True])) == "binary"

    def test_realistic_binary_outcome(self):
        rng = np.random.default_rng(0)
        Y = rng.binomial(1, 0.3, size=200)
        assert infer_outcome_type(Y) == "binary"


class TestContinuousDetection:
    def test_float(self):
        assert infer_outcome_type(np.array([0.5, 1.7, 2.3])) == "continuous"

    def test_int_negative_positive(self):
        # Bipolar {-1, +1} is not binary by our value-set rule.
        assert infer_outcome_type(np.array([-1, 1, -1])) == "continuous"

    def test_int_multiple_values(self):
        assert infer_outcome_type(np.array([0, 1, 2])) == "continuous"

    def test_float_in_unit_interval(self):
        # Values in [0, 1] but not in {0, 1}: probabilities, continuous.
        assert infer_outcome_type(np.array([0.0, 0.5, 1.0])) == "continuous"

    def test_float_two_distinct_non_binary_values(self):
        # Two unique values that aren't {0, 1}: continuous.
        assert infer_outcome_type(np.array([3.5, 4.7])) == "continuous"

    def test_negative_continuous(self):
        assert infer_outcome_type(np.array([-2.5, 0.0, 3.7])) == "continuous"

    def test_realistic_continuous_outcome(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=200)
        assert infer_outcome_type(Y) == "continuous"


class TestRejection:
    def test_string_array(self):
        with pytest.raises(ValueError, match="non-numeric"):
            infer_outcome_type(np.array(["yes", "no", "yes"]))

    def test_object_array_mixed(self):
        with pytest.raises(ValueError, match="non-numeric"):
            infer_outcome_type(np.array([1, "two", 3], dtype=object))

    def test_pandas_categorical_string(self):
        with pytest.raises(ValueError, match="non-numeric"):
            infer_outcome_type(pd.Categorical(["a", "b", "a"]))

    def test_nan_in_float_array(self):
        with pytest.raises(ValueError, match="NaN"):
            infer_outcome_type(np.array([0.0, 1.0, np.nan]))

    def test_nan_in_pandas_series(self):
        with pytest.raises(ValueError, match="NaN"):
            infer_outcome_type(pd.Series([0.0, 1.0, np.nan]))

    def test_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            infer_outcome_type(np.array([]))

    def test_single_class_all_zeros(self):
        # All-zero (or any constant) Y is degenerate: rejected, not binary.
        with pytest.raises(ValueError, match="one distinct value"):
            infer_outcome_type(np.array([0, 0, 0]))

    def test_single_class_all_ones(self):
        with pytest.raises(ValueError, match="one distinct value"):
            infer_outcome_type(np.array([1, 1, 1]))

    def test_single_value_continuous(self):
        # A single repeated non-0/1 value is also rejected.
        with pytest.raises(ValueError, match="one distinct value"):
            infer_outcome_type(np.array([3.5, 3.5, 3.5]))
