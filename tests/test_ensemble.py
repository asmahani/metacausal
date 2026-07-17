"""Tests for CausalEnsemble using mock estimators."""

import numpy as np
import pytest

from metacausal import (
    AteEstimate,
    BootstrapResult,
    CateEstimate,
    CausalEnsemble,
    CausalEstimate,
    ComponentAteEstimate,
    ComponentCateEstimate,
    ComponentFailureWarning,
    EnsembleEstimate,
    Mean,
    Median,
)
from metacausal.adapters import GenericAdapter, GenericCATEAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEstimator:
    """A mock causal estimator that returns a fixed ATE."""

    def __init__(self, ate: float, name: str = "mock"):
        self._ate = ate
        self._name = name
        self._is_fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, **kwargs):
        self._is_fitted = True

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._ate)

    def cate(self, X):
        raise NotImplementedError("MockEstimator does not support CATE.")

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class NoisyEstimator:
    """A mock estimator whose ATE depends on random_state."""

    def __init__(self, name: str = "noisy"):
        self._name = name
        self._ate_value: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, **kwargs):
        rs = kwargs.get("random_state")
        rng = np.random.default_rng(rs)
        self._ate_value = float(rng.normal())

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._ate_value)

    def cate(self, X):
        raise NotImplementedError("NoisyEstimator does not support CATE.")

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class FailingEstimator:
    """A mock estimator that always raises during fit."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, **kwargs):
        raise RuntimeError("Intentional failure")

    def ate(self, X=None):
        raise RuntimeError("Not fitted")

    def cate(self, X):
        raise RuntimeError("Not fitted")

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class CateCapableMock:
    """A mock estimator that reports supports_cate=True and produces CATE."""

    def __init__(self, ate: float, cate_offset: float = 0.0, name: str = "cate_mock"):
        self._ate = ate
        self._cate_offset = cate_offset
        self._name = name
        self._n: int | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, **kwargs):
        self._n = X.shape[0]

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._ate)

    def cate(self, X):
        n = X.shape[0]
        return ComponentCateEstimate(
            cate=np.full(n, self._ate + self._cate_offset)
        )

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class NoisyCateMock:
    """A CATE mock whose output varies with the bootstrap resample.

    Returns CATE = data_mean(Y) * ones(n) so the point estimate on original
    data differs from the mean of bootstrap replicates.
    """

    def __init__(self, name: str = "noisy_cate"):
        self._name = name
        self._mean_y: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, **kwargs):
        self._mean_y = float(np.mean(Y))

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._mean_y)

    def cate(self, X):
        n = X.shape[0]
        return ComponentCateEstimate(cate=np.full(n, self._mean_y))

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


def _dummy_data(n=100, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    T = rng.binomial(1, 0.5, size=n)
    Y = T * 2.0 + rng.normal(size=n)
    return X, T, Y


# ---------------------------------------------------------------------------
# Tests: basic aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_median_odd(self):
        methods = [
            MockEstimator(1.0, "a"),
            MockEstimator(3.0, "b"),
            MockEstimator(5.0, "c"),
        ]
        ens = CausalEnsemble(methods, aggregation="median")
        result = ens.estimate(*_dummy_data())
        assert result.ate == 3.0
        assert result.aggregation == "Median"

    def test_median_even(self):
        methods = [
            MockEstimator(1.0, "a"),
            MockEstimator(3.0, "b"),
            MockEstimator(5.0, "c"),
            MockEstimator(7.0, "d"),
        ]
        ens = CausalEnsemble(methods, aggregation="median")
        result = ens.estimate(*_dummy_data())
        assert result.ate == 4.0

    def test_mean(self):
        methods = [
            MockEstimator(1.0, "a"),
            MockEstimator(3.0, "b"),
            MockEstimator(5.0, "c"),
        ]
        ens = CausalEnsemble(methods, aggregation="mean")
        result = ens.estimate(*_dummy_data())
        assert result.ate == 3.0
        assert result.aggregation == "Mean"

    def test_median_robust_to_outlier(self):
        methods = [
            MockEstimator(2.0, "a"),
            MockEstimator(2.1, "b"),
            MockEstimator(2.2, "c"),
            MockEstimator(2.3, "d"),
            MockEstimator(100.0, "outlier"),
        ]
        ens = CausalEnsemble(methods, aggregation="median")
        result = ens.estimate(*_dummy_data())
        assert result.ate == 2.2

    def test_strategy_object_median(self):
        """CausalEnsemble(aggregation=Median()) and aggregation='median' are identical."""
        methods = [MockEstimator(1.0, "a"), MockEstimator(3.0, "b")]
        data = _dummy_data()
        r_str = CausalEnsemble(methods, aggregation="median").estimate(*data)
        r_obj = CausalEnsemble(methods, aggregation=Median()).estimate(*data)
        assert r_str.ate == r_obj.ate
        assert r_str.aggregation == r_obj.aggregation

    def test_strategy_object_mean(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(3.0, "b")]
        data = _dummy_data()
        r_str = CausalEnsemble(methods, aggregation="mean").estimate(*data)
        r_obj = CausalEnsemble(methods, aggregation=Mean()).estimate(*data)
        assert r_str.ate == r_obj.ate

    def test_invalid_aggregation_string(self):
        with pytest.raises(ValueError, match="aggregation"):
            CausalEnsemble([MockEstimator(1.0)], aggregation="trimmed")


# ---------------------------------------------------------------------------
# Tests: component estimates
# ---------------------------------------------------------------------------


class TestComponentEstimates:
    def test_component_ates(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data())
        assert result.component_ates == {"a": 1.0, "b": 3.0}

    def test_spread(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(5.0, "b")]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data())
        assert result.spread == 4.0

    def test_n_methods(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(2.0, "b")]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data())
        assert result.n_methods == 2

    def test_method_names(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(2.0, "b")]
        ens = CausalEnsemble(methods)
        assert ens.method_names == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests: failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_one_method_fails(self):
        methods = [
            MockEstimator(2.0, "good"),
            FailingEstimator(),
        ]
        ens = CausalEnsemble(methods)
        with pytest.warns(ComponentFailureWarning, match="failing"):
            result = ens.estimate(*_dummy_data())
        assert result.ate == 2.0
        assert result.n_methods == 1

    def test_all_methods_fail(self):
        ens = CausalEnsemble([FailingEstimator()])
        with pytest.raises(RuntimeError, match="All component methods"):
            ens.estimate(*_dummy_data())


# ---------------------------------------------------------------------------
# Tests: bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_bootstrap_returns_bootstrap_result(self):
        methods = [MockEstimator(2.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data(), n_boot=50, random_state=42)
        assert isinstance(result, BootstrapResult)

    def test_bootstrap_produces_ci(self):
        methods = [MockEstimator(2.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data(), n_boot=50, random_state=42)
        assert result.ate_ci_lower is not None
        assert result.ate_ci_upper is not None
        assert result.ate_ci_lower <= result.ate <= result.ate_ci_upper
        assert result.boot_ates is not None
        assert len(result.boot_ates) == 50

    def test_bootstrap_component_ates(self):
        methods = [MockEstimator(2.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data(), n_boot=10, random_state=42)
        assert isinstance(result, BootstrapResult)
        assert set(result.component_boot_ates.keys()) == {"a", "b"}
        assert len(result.component_boot_ates["a"]) == 10
        assert len(result.component_boot_ates["b"]) == 10

    def test_no_bootstrap_by_default(self):
        methods = [MockEstimator(2.0, "a")]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data())
        assert isinstance(result, AteEstimate)

    def test_bootstrap_method_directly(self):
        """fit() then bootstrap() produces BootstrapResult."""
        methods = [MockEstimator(2.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        ens.fit(*_dummy_data(), random_state=42)
        result = ens.bootstrap(n_boot=20, random_state=42)
        assert isinstance(result, BootstrapResult)
        assert result.n_boot == 20
        assert len(result.boot_ates) == 20

    def test_bootstrap_metadata(self):
        methods = [MockEstimator(2.0, "a")]
        ens = CausalEnsemble(methods)
        ens.fit(*_dummy_data(), random_state=0)
        result = ens.bootstrap(n_boot=10, alpha=0.1, random_state=0)
        assert result.alpha == 0.1
        assert result.aggregation == "Median"
        assert result.ensemble_weights is None  # pointwise strategy

    def test_bootstrap_cate(self):
        """bootstrap() computes CATE CIs when CATE-capable adapters are present."""
        methods = [CateCapableMock(2.0, name="a"), CateCapableMock(3.0, name="b")]
        X, T, Y = _dummy_data()
        ens = CausalEnsemble(methods)
        ens.fit(X, T, Y, random_state=0)
        result = ens.bootstrap(X, n_boot=10, random_state=0)
        assert result.cate is not None
        assert result.cate.shape == (X.shape[0],)
        assert result.cate_ci_lower is not None
        assert result.cate_ci_upper is not None
        assert result.boot_cates is not None
        assert result.boot_cates.shape == (10, X.shape[0])

    def test_bootstrap_cate_is_from_original_fit_not_replicate_mean(self):
        """BootstrapResult.cate must equal the point estimate from the original fit.

        NoisyCateMock returns mean(Y_train) as its CATE. Bootstrap replicates
        use resampled data, so their mean(Y_boot) differs from mean(Y_train).
        Verifying result.cate == original cate(X) (not boot_cates.mean(axis=0))
        confirms the point estimate comes from the original fit.
        """
        X, T, Y = _dummy_data(n=200, seed=0)
        ens = CausalEnsemble([NoisyCateMock()], aggregation="median")
        ens.fit(X, T, Y, random_state=0)
        original_cate = ens.cate(X).cate
        result = ens.bootstrap(X, n_boot=50, random_state=0)
        np.testing.assert_array_equal(result.cate, original_cate)
        # Sanity: bootstrap mean should differ from the original (resampling moves the mean)
        assert not np.allclose(result.boot_cates.mean(axis=0), original_cate)

    def test_bootstrap_no_cate_adapters(self):
        """bootstrap() sets CATE fields to None when no CATE-capable adapters."""
        methods = [MockEstimator(2.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        ens.fit(*_dummy_data(), random_state=0)
        result = ens.bootstrap(n_boot=10, random_state=0)
        assert result.cate is None
        assert result.cate_ci_lower is None
        assert result.cate_ci_upper is None
        assert result.boot_cates is None

    def test_bootstrap_populates_component_ate_estimates(self):
        """component_ate_estimates carries full-sample per-component point ATEs."""
        methods = [MockEstimator(2.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        ens.fit(*_dummy_data(), random_state=0)
        result = ens.bootstrap(n_boot=5, random_state=0)
        assert set(result.component_ate_estimates.keys()) == {"a", "b"}
        for name, est in result.component_ate_estimates.items():
            assert isinstance(est, ComponentAteEstimate)
        assert result.component_ate_estimates["a"].ate == 2.0
        assert result.component_ate_estimates["b"].ate == 3.0

    def test_bootstrap_populates_component_cate_estimates(self):
        """component_cate_estimates populated when CATE-capable adapters exist."""
        methods = [CateCapableMock(2.0, name="a"), CateCapableMock(3.0, name="b")]
        X, T, Y = _dummy_data()
        ens = CausalEnsemble(methods)
        ens.fit(X, T, Y, random_state=0)
        result = ens.bootstrap(X, n_boot=5, random_state=0)
        assert result.component_cate_estimates is not None
        assert set(result.component_cate_estimates.keys()) == {"a", "b"}
        for est in result.component_cate_estimates.values():
            assert isinstance(est, ComponentCateEstimate)
            assert est.cate.shape == (X.shape[0],)
        np.testing.assert_allclose(
            result.component_cate_estimates["a"].cate, 2.0
        )

    def test_bootstrap_component_cate_estimates_none_without_cate(self):
        """component_cate_estimates is None when no adapter supports CATE."""
        methods = [MockEstimator(2.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods)
        ens.fit(*_dummy_data(), random_state=0)
        result = ens.bootstrap(n_boot=5, random_state=0)
        assert result.component_cate_estimates is None


class TestReproducibility:
    def test_point_estimate_reproducible(self):
        methods = [NoisyEstimator("a"), NoisyEstimator("b")]
        data = _dummy_data()
        ens = CausalEnsemble(methods)
        r1 = ens.estimate(*data, random_state=99)
        r2 = ens.estimate(*data, random_state=99)
        assert r1.ate == r2.ate
        assert r1.component_ates == r2.component_ates

    def test_bootstrap_reproducible(self):
        methods = [NoisyEstimator("a"), NoisyEstimator("b")]
        data = _dummy_data()
        ens = CausalEnsemble(methods)
        r1 = ens.estimate(*data, n_boot=20, random_state=7)
        r2 = ens.estimate(*data, n_boot=20, random_state=7)
        assert r1.boot_ates.tolist() == r2.boot_ates.tolist()

    def test_different_seeds_differ(self):
        methods = [NoisyEstimator("a")]
        data = _dummy_data()
        ens = CausalEnsemble(methods)
        r1 = ens.estimate(*data, random_state=1)
        r2 = ens.estimate(*data, random_state=2)
        assert r1.ate != r2.ate

    def test_bootstrap_idempotent(self):
        """Calling bootstrap() twice on same fitted object gives same result."""
        methods = [NoisyEstimator("a"), NoisyEstimator("b")]
        ens = CausalEnsemble(methods)
        ens.fit(*_dummy_data(), random_state=42)
        r1 = ens.bootstrap(n_boot=20, random_state=5)
        r2 = ens.bootstrap(n_boot=20, random_state=5)
        assert r1.boot_ates.tolist() == r2.boot_ates.tolist()


# ---------------------------------------------------------------------------
# Tests: auto-wrapping
# ---------------------------------------------------------------------------


class TestAutoWrap:
    def test_callable_wrapped(self):
        fn = lambda X, T, Y: 42.0
        ens = CausalEnsemble([fn])
        result = ens.estimate(*_dummy_data())
        assert result.ate == 42.0

    def test_callable_pair_wrapped_as_generic_cate(self):
        def fn_fit(X, T, Y, **kwargs):
            return 2.0

        def fn_cate(state, X):
            return np.full(len(X), state)

        ens = CausalEnsemble([(fn_fit, fn_cate)])
        ens.fit(*_dummy_data())

        adapter = ens._fitted_adapters[0]
        assert isinstance(adapter, GenericCATEAdapter)
        np.testing.assert_allclose(ens.cate(_dummy_data()[0]).cate, 2.0)

    def test_callable_triple_wrapped_as_generic_cate(self):
        def fn_fit(X, T, Y, **kwargs):
            return 2.0

        def fn_cate(state, X):
            return np.full(len(X), state)

        def fn_ate(state, X):
            return state + 1.0

        ens = CausalEnsemble([(fn_fit, fn_cate, fn_ate)])
        ens.fit(*_dummy_data())

        adapter = ens._fitted_adapters[0]
        assert isinstance(adapter, GenericCATEAdapter)
        assert ens.ate().ate == 3.0

    def test_generic_adapter(self):
        adapter = GenericAdapter(lambda X, T, Y: 7.0, name="my_fn")
        ens = CausalEnsemble([adapter])
        result = ens.estimate(*_dummy_data())
        assert result.ate == 7.0
        assert "my_fn" in result.component_ates

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Cannot wrap"):
            CausalEnsemble(["not_an_estimator"])


# ---------------------------------------------------------------------------
# Tests: fit/predict API
# ---------------------------------------------------------------------------


class TestFitPredict:
    def test_fit_returns_self(self):
        ens = CausalEnsemble([MockEstimator(1.0, "a")])
        result = ens.fit(*_dummy_data())
        assert result is ens

    def test_fit_then_ate(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(3.0, "b")]
        ens = CausalEnsemble(methods, aggregation="median")
        ens.fit(*_dummy_data(), random_state=42)
        result = ens.ate()
        assert result.ate == 2.0
        assert isinstance(result, AteEstimate)

    def test_chaining(self):
        methods = [MockEstimator(5.0, "a")]
        result = CausalEnsemble(methods).fit(*_dummy_data()).ate()
        assert result.ate == 5.0

    def test_ate_before_fit_raises(self):
        ens = CausalEnsemble([MockEstimator(1.0, "a")])
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.ate()

    def test_fit_ate_matches_estimate(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(3.0, "b")]
        data = _dummy_data()
        ens = CausalEnsemble(methods)
        est_result = ens.estimate(*data, random_state=42)
        ens2 = CausalEnsemble(methods)
        ens2.fit(*data, random_state=42)
        fit_result = ens2.ate()
        assert est_result.ate == fit_result.ate
        assert est_result.component_ates == fit_result.component_ates

    def test_cate_method_names(self):
        methods = [
            MockEstimator(1.0, "ate_only"),
            CateCapableMock(2.0, name="cate_ok"),
        ]
        ens = CausalEnsemble(methods)
        assert ens.method_names == ["ate_only", "cate_ok"]
        assert ens.cate_method_names == ["cate_ok"]

    def test_component_fit_times(self):
        methods = [MockEstimator(1.0, "a")]
        ens = CausalEnsemble(methods)
        ens.fit(*_dummy_data())
        result = ens.ate()
        assert result.component_fit_times is not None
        assert "a" in result.component_fit_times
        assert result.component_fit_times["a"] >= 0


# ---------------------------------------------------------------------------
# Tests: CATE
# ---------------------------------------------------------------------------


class TestCate:
    def test_cate_basic(self):
        methods = [CateCapableMock(2.0, name="a")]
        ens = CausalEnsemble(methods)
        X, T, Y = _dummy_data()
        ens.fit(X, T, Y)
        result = ens.cate(X)
        assert isinstance(result, CateEstimate)
        assert result.cate.shape == (X.shape[0],)
        np.testing.assert_allclose(result.cate, 2.0)

    def test_cate_no_x_uses_train(self):
        """cate() with X=None uses training data."""
        methods = [CateCapableMock(2.0, name="a")]
        X, T, Y = _dummy_data()
        ens = CausalEnsemble(methods)
        ens.fit(X, T, Y)
        result_explicit = ens.cate(X)
        result_implicit = ens.cate()
        np.testing.assert_allclose(result_explicit.cate, result_implicit.cate)

    def test_cate_median_aggregation(self):
        methods = [
            CateCapableMock(1.0, name="a"),
            CateCapableMock(3.0, name="b"),
            CateCapableMock(5.0, name="c"),
        ]
        ens = CausalEnsemble(methods, aggregation="median")
        X, T, Y = _dummy_data()
        ens.fit(X, T, Y)
        result = ens.cate(X)
        np.testing.assert_allclose(result.cate, 3.0)
        assert result.n_methods == 3

    def test_cate_mean_aggregation(self):
        methods = [
            CateCapableMock(1.0, name="a"),
            CateCapableMock(3.0, name="b"),
            CateCapableMock(5.0, name="c"),
        ]
        ens = CausalEnsemble(methods, aggregation="mean")
        X, T, Y = _dummy_data()
        ens.fit(X, T, Y)
        result = ens.cate(X)
        np.testing.assert_allclose(result.cate, 3.0)

    def test_cate_excludes_non_cate_methods(self):
        methods = [
            MockEstimator(10.0, "ate_only"),
            CateCapableMock(2.0, name="cate_a"),
            CateCapableMock(4.0, name="cate_b"),
        ]
        ens = CausalEnsemble(methods)
        X, T, Y = _dummy_data()
        ens.fit(X, T, Y)
        result = ens.cate(X)
        assert set(result.component_estimates.keys()) == {"cate_a", "cate_b"}
        assert result.n_methods == 2
        np.testing.assert_allclose(result.cate, 3.0)

    def test_cate_no_capable_methods_raises(self):
        methods = [MockEstimator(1.0, "a"), MockEstimator(2.0, "b")]
        ens = CausalEnsemble(methods)
        X, T, Y = _dummy_data()
        ens.fit(X, T, Y)
        with pytest.raises(RuntimeError, match="No component methods support CATE"):
            ens.cate(X)

    def test_cate_before_fit_raises(self):
        methods = [CateCapableMock(1.0, name="a")]
        ens = CausalEnsemble(methods)
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.cate(np.zeros((10, 5)))

    def test_cate_component_cates_property(self):
        methods = [
            CateCapableMock(1.0, name="a"),
            CateCapableMock(3.0, name="b"),
        ]
        ens = CausalEnsemble(methods)
        X, T, Y = _dummy_data()
        ens.fit(X, T, Y)
        result = ens.cate(X)
        comp = result.component_cates
        assert set(comp.keys()) == {"a", "b"}
        np.testing.assert_allclose(comp["a"], 1.0)
        np.testing.assert_allclose(comp["b"], 3.0)

    def test_cate_ensemble_weights_none_for_pointwise(self):
        methods = [CateCapableMock(1.0, name="a"), CateCapableMock(2.0, name="b")]
        X, T, Y = _dummy_data()
        result = CausalEnsemble(methods).fit(X, T, Y).cate(X)
        assert result.ensemble_weights is None

    def test_cate_chaining(self):
        methods = [CateCapableMock(5.0, name="a")]
        X, T, Y = _dummy_data()
        result = CausalEnsemble(methods).fit(X, T, Y).cate(X)
        np.testing.assert_allclose(result.cate, 5.0)

    def test_cate_different_test_size(self):
        methods = [CateCapableMock(2.0, name="a")]
        X_train, T, Y = _dummy_data(n=100)
        X_test = np.zeros((20, 5))
        ens = CausalEnsemble(methods)
        ens.fit(X_train, T, Y)
        result = ens.cate(X_test)
        assert result.cate.shape == (20,)


# ---------------------------------------------------------------------------
# Tests: backward-compatible type aliases
# ---------------------------------------------------------------------------


class TestTypeAliases:
    def test_causal_estimate_alias(self):
        assert CausalEstimate is ComponentAteEstimate

    def test_ensemble_estimate_alias(self):
        assert EnsembleEstimate is AteEstimate


# ---------------------------------------------------------------------------
# Tests: duplicate adapter name detection (issue #14)
# ---------------------------------------------------------------------------


class TestDuplicateNames:
    def test_duplicate_names_raises(self):
        """Two adapters with the same name raise ValueError at construction."""
        with pytest.raises(ValueError, match="Duplicate adapter name"):
            CausalEnsemble([
                MockEstimator(1.0, "same"),
                MockEstimator(2.0, "same"),
            ])

    def test_three_way_duplicate_raises(self):
        """Three adapters sharing a name also raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate adapter name"):
            CausalEnsemble([
                MockEstimator(1.0, "x"),
                MockEstimator(2.0, "x"),
                MockEstimator(3.0, "x"),
            ])

    def test_explicit_names_disambiguate(self):
        """Explicitly distinct names bypass the duplicate check."""
        ens = CausalEnsemble([
            MockEstimator(1.0, "m_1"),
            MockEstimator(2.0, "m_2"),
        ])
        result = ens.estimate(*_dummy_data())
        assert set(result.component_ates.keys()) == {"m_1", "m_2"}
        assert len(result.component_estimates) == 2

    def test_no_duplicates_unchanged(self):
        """All-distinct names: behaviour is unchanged, no error raised."""
        methods = [MockEstimator(float(i), str(i)) for i in range(5)]
        ens = CausalEnsemble(methods)
        result = ens.estimate(*_dummy_data())
        assert len(result.component_estimates) == 5
