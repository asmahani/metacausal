"""Tests for TrimmedMean aggregation strategy."""

from __future__ import annotations

import numpy as np
import pytest

from metacausal import CausalEnsemble, TrimmedMean
from metacausal.aggregation import PointwiseStrategy
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEstimator:
    """ATE-only mock estimator."""

    def __init__(self, ate_value: float, name: str = "mock"):
        self._ate = ate_value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, **kwargs):
        pass

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._ate)

    def cate(self, X):
        raise NotImplementedError

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class CateCapableMock:
    """Mock estimator with CATE support."""

    def __init__(self, cate_values: np.ndarray, name: str = "mock"):
        self._cate_values = np.asarray(cate_values, dtype=float)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, **kwargs):
        pass

    def ate(self, X=None):
        return ComponentAteEstimate(ate=float(self._cate_values.mean()))

    def cate(self, X):
        n = X.shape[0]
        return ComponentCateEstimate(cate=self._cate_values[:n].copy())

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


def _dummy_data(n=50, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    T = rng.binomial(1, 0.5, size=n)
    Y = T * 1.0 + rng.normal(size=n)
    return X, T, Y


# ---------------------------------------------------------------------------
# Tests: TrimmedMean.aggregate — 1-D (scalar ATE) input
# ---------------------------------------------------------------------------


class TestTrimmedMeanAggregate1D:
    def test_default_trim_count_k5(self):
        """K=5, trim_count=1: drops min and max, averages middle 3."""
        tm = TrimmedMean()
        ates = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = tm.aggregate(ates)
        np.testing.assert_allclose(result, (2.0 + 3.0 + 4.0) / 3)

    def test_symmetric_trim(self):
        """Both tails are trimmed: outlier at bottom is also dropped."""
        tm = TrimmedMean(trim_count=1)
        ates = np.array([-100.0, 2.0, 3.0, 4.0, 5.0])
        result = tm.aggregate(ates)
        np.testing.assert_allclose(result, (2.0 + 3.0 + 4.0) / 3)

    def test_trim_count_2(self):
        """trim_count=2 with K=7: drops bottom 2 and top 2."""
        tm = TrimmedMean(trim_count=2)
        ates = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = tm.aggregate(ates)
        np.testing.assert_allclose(result, (2.0 + 3.0 + 4.0) / 3)

    def test_order_invariant(self):
        """Result is independent of input order."""
        tm = TrimmedMean()
        ates = np.array([3.0, 1.0, 100.0, 2.0, 4.0])
        result = tm.aggregate(ates)
        np.testing.assert_allclose(result, (2.0 + 3.0 + 4.0) / 3)

    def test_trim_too_aggressive_raises(self):
        """2 * trim_count >= K raises ValueError."""
        tm = TrimmedMean(trim_count=2)
        with pytest.raises(ValueError, match="trim_count"):
            tm.aggregate(np.array([1.0, 2.0, 3.0]))  # K=3, 2*2=4 >= 3

    def test_trim_count_equals_half_k_raises(self):
        """trim_count = K//2 with even K raises ValueError."""
        tm = TrimmedMean(trim_count=2)
        with pytest.raises(ValueError, match="trim_count"):
            tm.aggregate(np.array([1.0, 2.0, 3.0, 4.0]))  # K=4, 2*2=4 >= 4

    def test_all_equal_values(self):
        """All component ATEs equal: trimmed mean equals that value."""
        tm = TrimmedMean()
        ates = np.full(5, 3.14)
        np.testing.assert_allclose(tm.aggregate(ates), 3.14)

    def test_minimum_k3(self):
        """K=3 with trim_count=1: only middle value remains."""
        tm = TrimmedMean(trim_count=1)
        ates = np.array([1.0, 5.0, 100.0])
        np.testing.assert_allclose(tm.aggregate(ates), 5.0)


# ---------------------------------------------------------------------------
# Tests: TrimmedMean.aggregate — 2-D (CATE matrix) input
# ---------------------------------------------------------------------------


class TestTrimmedMeanAggregate2D:
    def test_trims_per_sample(self):
        """Each sample trims the top and bottom model value."""
        tm = TrimmedMean(trim_count=1)
        # K=3 models, n=2 samples
        # sample 0: models predict [1, 2, 100] → mean([2]) = 2.0
        # sample 1: models predict [5, 3, 4]   → mean([4]) = 4.0
        cate_matrix = np.array([[1.0, 5.0], [2.0, 3.0], [100.0, 4.0]])
        result = tm.aggregate(cate_matrix)
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_output_shape(self):
        """aggregate returns shape (n,)."""
        tm = TrimmedMean()
        rng = np.random.default_rng(0)
        cate_matrix = rng.normal(size=(5, 30))
        result = tm.aggregate(cate_matrix)
        assert result.shape == (30,)

    def test_trim_too_aggressive_raises(self):
        """2 * trim_count >= K raises ValueError."""
        tm = TrimmedMean(trim_count=2)
        with pytest.raises(ValueError, match="trim_count"):
            tm.aggregate(np.ones((3, 10)))

    def test_all_equal_models(self):
        """All models identical: trimmed mean equals that value everywhere."""
        tm = TrimmedMean()
        cate_matrix = np.tile(np.arange(10, dtype=float), (5, 1))
        result = tm.aggregate(cate_matrix)
        np.testing.assert_allclose(result, np.arange(10, dtype=float))


# ---------------------------------------------------------------------------
# Tests: TrimmedMean is a PointwiseStrategy
# ---------------------------------------------------------------------------


class TestTrimmedMeanProtocol:
    def test_is_pointwise_strategy(self):
        assert isinstance(TrimmedMean(), PointwiseStrategy)

    def test_strategy_family(self):
        assert TrimmedMean.strategy_family == "pointwise"

    def test_ensemble_weights_is_none(self):
        assert TrimmedMean().ensemble_weights is None


# ---------------------------------------------------------------------------
# Tests: CausalEnsemble with TrimmedMean
# ---------------------------------------------------------------------------


class TestCausalEnsembleTrimmedMean:
    def test_string_alias(self):
        """CausalEnsemble(aggregation='trimmed_mean') resolves to TrimmedMean."""
        X, T, Y = _dummy_data()
        n = X.shape[0]
        ens = CausalEnsemble(
            methods=[MockEstimator(1.0, name="m0")],
            aggregation="trimmed_mean",
        )
        assert isinstance(ens.aggregation, TrimmedMean)

    def test_ate_trims_outlier(self):
        """ATE ignores outlier estimator after trimming."""
        X, T, Y = _dummy_data()
        methods = [
            MockEstimator(2.0, name="m0"),
            MockEstimator(3.0, name="m1"),
            MockEstimator(4.0, name="m2"),
            MockEstimator(5.0, name="m3"),
            MockEstimator(100.0, name="outlier"),
        ]
        ens = CausalEnsemble(methods=methods, aggregation=TrimmedMean(trim_count=1))
        ens.fit(X, T, Y)
        result = ens.ate(X)
        np.testing.assert_allclose(result.ate, (3.0 + 4.0 + 5.0) / 3)

    def test_cate_trims_outlier(self):
        """CATE ignores outlier model after trimming."""
        X, T, Y = _dummy_data(n=50)
        n = X.shape[0]
        methods = [
            CateCapableMock(np.full(n, 1.0), name="m0"),
            CateCapableMock(np.full(n, 2.0), name="m1"),
            CateCapableMock(np.full(n, 3.0), name="m2"),
            CateCapableMock(np.full(n, 4.0), name="m3"),
            CateCapableMock(np.full(n, 999.0), name="outlier"),
        ]
        ens = CausalEnsemble(methods=methods, aggregation=TrimmedMean(trim_count=1))
        ens.fit(X, T, Y)
        result = ens.cate(X)
        np.testing.assert_allclose(result.cate, (2.0 + 3.0 + 4.0) / 3)

    def test_aggregation_label_in_result(self):
        """AteEstimate.aggregation reflects TrimmedMean."""
        X, T, Y = _dummy_data()
        methods = [MockEstimator(float(i), name=f"m{i}") for i in range(5)]
        ens = CausalEnsemble(methods=methods, aggregation="trimmed_mean")
        ens.fit(X, T, Y)
        result = ens.ate(X)
        assert result.aggregation == "TrimmedMean"
