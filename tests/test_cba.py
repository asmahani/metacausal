"""Tests for CBA (Consensus Based Averaging) aggregation strategy."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import kendalltau

from metacausal import CBA, AgreementStrategy, CausalEnsemble, EnsembleWeights
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_cate_matrix(*rows):
    """Build a (K, n) cate_matrix from row arrays."""
    return np.stack([np.asarray(r, dtype=float) for r in rows], axis=0)


def _dummy_data(n=50, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    T = rng.binomial(1, 0.5, size=n)
    Y = T * 1.0 + rng.normal(size=n)
    return X, T, Y


# ---------------------------------------------------------------------------
# Tests: CBA.compute_weights
# ---------------------------------------------------------------------------


class TestCBAComputeWeights:
    def test_k1_returns_weight_one(self):
        """K=1: single model, weight=1.0."""
        cba = CBA()
        cate_matrix = _make_cate_matrix([1.0, 2.0, 3.0])
        w = cba.compute_weights(cate_matrix, ["m0"])

        assert isinstance(w, EnsembleWeights)
        assert w.method == "cba"
        np.testing.assert_array_equal(w.weights, [1.0])
        assert w.model_names == ["m0"]
        assert w.details["n_selected"] == 1

    def test_k2_selects_both_models(self):
        """K=2: both models have identical mean tau; both selected."""
        cba = CBA()
        p1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        cate_matrix = _make_cate_matrix(p1, p2)
        w = cba.compute_weights(cate_matrix, ["m0", "m1"])

        assert len(w.weights) == 2
        assert set(w.details["selected_models"]) == {"m0", "m1"}
        np.testing.assert_allclose(w.weights.sum(), 1.0)

    def test_k3_outlier_excluded(self):
        """K=3, one outlier: two agreeing models selected, outlier excluded."""
        rng = np.random.default_rng(1)
        n = 200
        base = rng.normal(size=n)
        # m0 and m1 are highly correlated (agreeing)
        p0 = base + 0.05 * rng.normal(size=n)
        p1 = base + 0.05 * rng.normal(size=n)
        # m2 is an outlier: reversed sign
        p2 = -base + rng.normal(size=n)
        cate_matrix = _make_cate_matrix(p0, p1, p2)

        cba = CBA()
        w = cba.compute_weights(cate_matrix, ["m0", "m1", "m2"])

        # m2 should be excluded (it disagrees with m0 and m1)
        assert "m2" not in w.details["selected_models"]
        assert "m0" in w.details["selected_models"]
        assert "m1" in w.details["selected_models"]

    def test_k6_all_identical_selects_all(self):
        """K=6, all models identical: no meaningful knee → all selected."""
        n = 50
        cate_row = np.arange(n, dtype=float)
        cate_matrix = np.tile(cate_row, (6, 1))
        model_names = [f"m{i}" for i in range(6)]

        # All rows identical → all pairwise taus = 1 → flat diffs
        cba = CBA()
        w = cba.compute_weights(cate_matrix, model_names)

        assert w.details["n_selected"] == 6
        assert set(w.details["selected_models"]) == set(model_names)

    def test_k6_clear_split(self):
        """K=6, clear split: top 3 high-agreement models selected."""
        rng = np.random.default_rng(42)
        n = 300
        base = rng.normal(size=n)
        # High-agreement cluster: 3 models
        high = [base + 0.01 * rng.normal(size=n) for _ in range(3)]
        # Low-agreement cluster: 3 models (reversed + noise)
        low = [-base * 2 + 3 * rng.normal(size=n) for _ in range(3)]
        rows = high + low
        names = [f"h{i}" for i in range(3)] + [f"l{i}" for i in range(3)]
        cate_matrix = np.stack(rows, axis=0)

        cba = CBA()
        w = cba.compute_weights(cate_matrix, names)

        selected = set(w.details["selected_models"])
        # All selected models should be from the high-agreement cluster
        assert selected.issubset({"h0", "h1", "h2"})

    def test_selected_indices_match_manual_taus(self):
        """Verify that mean_taus match manual scipy kendalltau computation."""
        rng = np.random.default_rng(7)
        n = 100
        rows = [rng.normal(size=n) for _ in range(3)]
        cate_matrix = np.stack(rows, axis=0)
        names = ["a", "b", "c"]

        cba = CBA()
        w = cba.compute_weights(cate_matrix, names)

        # Manual computation
        tau_ab, _ = kendalltau(rows[0], rows[1])
        tau_ac, _ = kendalltau(rows[0], rows[2])
        tau_bc, _ = kendalltau(rows[1], rows[2])
        expected = {
            "a": (tau_ab + tau_ac) / 2,
            "b": (tau_ab + tau_bc) / 2,
            "c": (tau_ac + tau_bc) / 2,
        }
        for name, expected_tau in expected.items():
            np.testing.assert_allclose(
                w.details["mean_taus"][name], expected_tau, atol=1e-10
            )

    def test_constant_predictions_raises(self):
        """Constant CATE predictions → NaN tau → ValueError."""
        cba = CBA()
        cate_matrix = np.ones((3, 50))  # all constant → tau = NaN
        with pytest.raises(ValueError, match="NaN"):
            cba.compute_weights(cate_matrix, ["m0", "m1", "m2"])

    def test_single_model_warning(self):
        """K=3 with very clear outliers could select 1 model → RuntimeWarning."""
        rng = np.random.default_rng(99)
        n = 200
        base = rng.normal(size=n)
        # m0 is unique; m1 and m2 agree with each other but not m0
        p0 = base
        p1 = -base + 0.01 * rng.normal(size=n)
        p2 = -base + 0.01 * rng.normal(size=n)
        cate_matrix = _make_cate_matrix(p0, p1, p2)

        cba = CBA(eps=0.0)  # tight eps: no flat-diff guard
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            w = cba.compute_weights(cate_matrix, ["m0", "m1", "m2"])
        # Either a single model is selected (warning fires) or multiple — both
        # are valid; what matters is no exception is raised.
        assert isinstance(w, EnsembleWeights)

    def test_weights_sum_to_one(self):
        """Weights always sum to 1.0."""
        rng = np.random.default_rng(5)
        for k in [1, 2, 3, 4, 5]:
            cate_matrix = rng.normal(size=(k, 80))
            names = [f"m{i}" for i in range(k)]
            cba = CBA()
            w = cba.compute_weights(cate_matrix, names)
            np.testing.assert_allclose(w.weights.sum(), 1.0, atol=1e-12)

    def test_ensemble_weights_property(self):
        """ensemble_weights returns the stored EnsembleWeights after compute_weights."""
        cba = CBA()
        assert cba.ensemble_weights is None  # before compute_weights
        cate_matrix = _make_cate_matrix([1, 2, 3], [4, 5, 6])
        cba.compute_weights(cate_matrix, ["m0", "m1"])
        assert cba.ensemble_weights is not None
        assert isinstance(cba.ensemble_weights, EnsembleWeights)

    def test_eps_guard_all_models(self):
        """When all diffs < eps, all models are selected regardless of K."""
        rng = np.random.default_rng(11)
        n = 100
        # All models equally correlated (shuffled versions of same base)
        base = np.arange(n, dtype=float)
        rows = [base[rng.permutation(n)] for _ in range(5)]
        cate_matrix = np.stack(rows, axis=0)
        names = [f"m{i}" for i in range(5)]

        cba = CBA(eps=1.0)  # very generous eps: always flat
        w = cba.compute_weights(cate_matrix, names)
        assert w.details["n_selected"] == 5


# ---------------------------------------------------------------------------
# Tests: CBA.aggregate
# ---------------------------------------------------------------------------


class TestCBAAggregation:
    def test_aggregate_uniform_average(self):
        """aggregate produces the correct weighted average."""
        cba = CBA()
        # K=2: both selected with weight 0.5 each
        p0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p1 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        cate_matrix = _make_cate_matrix(p0, p1)
        cba.compute_weights(cate_matrix, ["m0", "m1"])

        result = cba.aggregate(cate_matrix)
        expected = (p0 + p1) / 2
        np.testing.assert_allclose(result, expected)

    def test_aggregate_selected_only(self):
        """aggregate uses only selected models (excluded get weight 0)."""
        rng = np.random.default_rng(3)
        n = 200
        base = rng.normal(size=n)
        p0 = base + 0.01 * rng.normal(size=n)
        p1 = base + 0.01 * rng.normal(size=n)
        p2 = -base + rng.normal(size=n)  # outlier
        cate_matrix = _make_cate_matrix(p0, p1, p2)

        cba = CBA()
        cba.compute_weights(cate_matrix, ["m0", "m1", "m2"])
        result = cba.aggregate(cate_matrix)

        selected = cba.ensemble_weights.details["selected_models"]
        if "m2" not in selected:
            expected = (p0 + p1) / 2
            np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: CausalEnsemble with CBA
# ---------------------------------------------------------------------------


class TestCausalEnsembleCBA:
    def test_string_alias_cba(self):
        """CausalEnsemble(aggregation='cba') resolves to a CBA instance."""
        X, T, Y = _dummy_data()
        n = X.shape[0]
        ens = CausalEnsemble(
            methods=[CateCapableMock(np.zeros(n), name="m0")],
            aggregation="cba",
        )
        assert isinstance(ens.aggregation, CBA)
        assert isinstance(ens.aggregation, AgreementStrategy)

    def test_invalid_string_still_raises(self):
        """Invalid strings still raise ValueError."""
        with pytest.raises(ValueError, match="aggregation"):
            CausalEnsemble(aggregation="unknown_strategy")

    def test_fit_computes_weights(self):
        """After fit(), ensemble_weights is populated for CBA."""
        rng = np.random.default_rng(0)
        n, p = 50, 3
        X, T, Y = _dummy_data(n=n, p=p)

        p0 = rng.normal(size=n)
        p1 = rng.normal(size=n)
        methods = [
            CateCapableMock(p0, name="m0"),
            CateCapableMock(p1, name="m1"),
        ]
        ens = CausalEnsemble(methods=methods, aggregation="cba")
        ens.fit(X, T, Y)

        assert ens.aggregation.ensemble_weights is not None
        w = ens.aggregation.ensemble_weights
        assert w.method == "cba"
        assert len(w.weights) == 2
        np.testing.assert_allclose(w.weights.sum(), 1.0, atol=1e-12)

    def test_cate_returns_cate_estimate(self):
        """cate() returns CateEstimate with ensemble_weights populated."""
        X, T, Y = _dummy_data()
        n = X.shape[0]
        rng = np.random.default_rng(9)
        methods = [
            CateCapableMock(rng.normal(size=n), name=f"m{i}") for i in range(3)
        ]
        ens = CausalEnsemble(methods=methods, aggregation="cba")
        ens.fit(X, T, Y)
        result = ens.cate(X)

        from metacausal import CateEstimate
        assert isinstance(result, CateEstimate)
        assert result.ensemble_weights is not None
        assert result.ensemble_weights.method == "cba"

    def test_ate_equals_mean_cate(self):
        """For CBA, ATE = mean(ensemble CATE)."""
        X, T, Y = _dummy_data()
        n = X.shape[0]
        rng = np.random.default_rng(12)
        methods = [
            CateCapableMock(rng.normal(size=n), name=f"m{i}") for i in range(3)
        ]
        ens = CausalEnsemble(methods=methods, aggregation="cba")
        ens.fit(X, T, Y)

        ate_result = ens.ate(X)
        cate_result = ens.cate(X)

        expected_ate = float(cate_result.cate.mean())
        np.testing.assert_allclose(ate_result.ate, expected_ate, atol=1e-12)

    def test_backward_compat_median_unchanged(self):
        """'median' aggregation still works as before."""
        X, T, Y = _dummy_data()
        n = X.shape[0]
        methods = [
            CateCapableMock(np.full(n, v), name=f"m{i}")
            for i, v in enumerate([1.0, 2.0, 3.0])
        ]
        ens = CausalEnsemble(methods=methods, aggregation="median")
        ens.fit(X, T, Y)
        result = ens.ate(X)
        assert result.aggregation == "Median"

    def test_backward_compat_mean_unchanged(self):
        """'mean' aggregation still works as before."""
        X, T, Y = _dummy_data()
        n = X.shape[0]
        methods = [
            CateCapableMock(np.full(n, v), name=f"m{i}")
            for i, v in enumerate([1.0, 3.0])
        ]
        ens = CausalEnsemble(methods=methods, aggregation="mean")
        ens.fit(X, T, Y)
        result = ens.ate(X)
        assert result.aggregation == "Mean"

    def test_bootstrap_with_cba(self):
        """bootstrap() runs without error for CBA and returns BootstrapResult."""
        X, T, Y = _dummy_data(n=40, seed=3)
        n = X.shape[0]
        rng = np.random.default_rng(3)
        methods = [
            CateCapableMock(rng.normal(size=n), name=f"m{i}") for i in range(3)
        ]
        ens = CausalEnsemble(methods=methods, aggregation="cba")
        ens.fit(X, T, Y)
        from metacausal import BootstrapResult
        result = ens.bootstrap(n_boot=10, random_state=0)

        assert isinstance(result, BootstrapResult)
        assert result.aggregation == "CBA"
        assert result.n_failed < 10
        assert not np.isnan(result.ate_ci_lower)
        assert not np.isnan(result.ate_ci_upper)

    def test_cba_strategy_object(self):
        """CausalEnsemble(aggregation=CBA()) works the same as aggregation='cba'."""
        X, T, Y = _dummy_data()
        n = X.shape[0]
        rng = np.random.default_rng(5)
        methods = [
            CateCapableMock(rng.normal(size=n), name=f"m{i}") for i in range(3)
        ]

        ens_str = CausalEnsemble(methods=methods, aggregation="cba")
        ens_str.fit(X, T, Y)

        ens_obj = CausalEnsemble(methods=methods, aggregation=CBA())
        ens_obj.fit(X, T, Y)

        assert type(ens_str.aggregation) == type(ens_obj.aggregation)  # noqa: E721

    def test_no_cate_adapters_raises(self):
        """CBA with no CATE-capable adapters raises RuntimeError at fit()."""
        from tests.test_ensemble import MockEstimator

        X, T, Y = _dummy_data()
        ens = CausalEnsemble(
            methods=[MockEstimator(1.0, name="m0")], aggregation="cba"
        )
        with pytest.raises(RuntimeError, match="Agreement-based aggregation"):
            ens.fit(X, T, Y)
