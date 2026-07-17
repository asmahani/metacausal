"""Tests for GenericATEAdapter (B1.1 rename) and GenericCATEAdapter (B1.1)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from metacausal import CausalEnsemble, GenericAdapter, GenericATEAdapter, GenericCATEAdapter
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dgp(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = X[:, 0] + T * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, T, Y


def _tlearner_fns():
    """A T-learner implemented as fn_fit / fn_cate / fn_ate."""

    def fn_fit(X, T, Y, **kwargs):
        treated = T == 1
        m1 = LinearRegression().fit(X[treated], Y[treated])
        m0 = LinearRegression().fit(X[~treated], Y[~treated])
        return m1, m0

    def fn_cate(state, X):
        m1, m0 = state
        return m1.predict(X) - m0.predict(X)

    def fn_ate(state, X):
        m1, m0 = state
        return float((m1.predict(X) - m0.predict(X)).mean())

    return fn_fit, fn_cate, fn_ate


# ---------------------------------------------------------------------------
# GenericATEAdapter (rename)
# ---------------------------------------------------------------------------


class TestGenericATEAdapter:
    def test_supports_cate_false(self):
        a = GenericATEAdapter(lambda X, T, Y: 1.0)
        assert a.supports_cate is False

    def test_ate_returns_scalar(self):
        X, T, Y = _dgp()
        a = GenericATEAdapter(lambda X, T, Y: 2.5, name="test")
        a.fit(X, T, Y)
        result = a.ate()
        assert isinstance(result, ComponentAteEstimate)
        assert result.ate == pytest.approx(2.5)

    def test_ate_accepts_component_ate_estimate(self):
        X, T, Y = _dgp()
        a = GenericATEAdapter(
            lambda X, T, Y: ComponentAteEstimate(ate=3.0), name="test"
        )
        a.fit(X, T, Y)
        result = a.ate()
        assert result.ate == pytest.approx(3.0)

    def test_cate_raises_not_implemented(self):
        a = GenericATEAdapter(lambda X, T, Y: 1.0)
        a.fit(*_dgp())
        with pytest.raises(NotImplementedError):
            a.cate(np.zeros((5, 3)))

    def test_not_fitted_raises(self):
        a = GenericATEAdapter(lambda X, T, Y: 1.0)
        with pytest.raises(RuntimeError, match="not fitted"):
            a.ate()

    def test_name_default(self):
        assert GenericATEAdapter(lambda X, T, Y: 0.0).name == "custom"

    def test_generic_adapter_alias(self):
        """GenericAdapter is the backward-compatible alias for GenericATEAdapter."""
        assert GenericAdapter is GenericATEAdapter

    def test_generic_adapter_still_works_in_ensemble(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[GenericAdapter(lambda X, T, Y: 1.5, name="old_api")],
            aggregation="median",
        )
        ens.fit(X, T, Y)
        assert ens.ate().ate == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# GenericCATEAdapter — basic interface
# ---------------------------------------------------------------------------


class TestGenericCATEAdapterInterface:
    def test_supports_cate_true(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        a = GenericCATEAdapter(fn_fit, fn_cate)
        assert a.supports_cate is True

    def test_name_default(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        assert GenericCATEAdapter(fn_fit, fn_cate).name == "custom_cate"

    def test_name_custom(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        a = GenericCATEAdapter(fn_fit, fn_cate, name="t_learner")
        assert a.name == "t_learner"

    def test_not_fitted_cate_raises(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        a = GenericCATEAdapter(fn_fit, fn_cate)
        with pytest.raises(RuntimeError, match="not fitted"):
            a.cate(np.zeros((5, 3)))

    def test_not_fitted_ate_raises(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        a = GenericCATEAdapter(fn_fit, fn_cate)
        with pytest.raises(RuntimeError, match="not fitted"):
            a.ate()


# ---------------------------------------------------------------------------
# GenericCATEAdapter — fit / cate / ate
# ---------------------------------------------------------------------------


class TestGenericCATEAdapterPrediction:
    def test_cate_shape(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=80)
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(X, T, Y)
        result = a.cate(X)
        assert isinstance(result, ComponentCateEstimate)
        assert result.cate.shape == (80,)

    def test_cate_on_new_x(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=100)
        X_new = np.random.default_rng(99).normal(size=(30, 3))
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(X, T, Y)
        result = a.cate(X_new)
        assert result.cate.shape == (30,)

    def test_fn_cate_n1_output_squeezed(self):
        """fn_cate returning (n, 1) is automatically squeezed to (n,)."""
        def fn_fit(X, T, Y, **kw): return None
        def fn_cate(state, X): return np.ones((len(X), 1))  # (n, 1)
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(*_dgp(n=50))
        result = a.cate(np.zeros((20, 3)))
        assert result.cate.shape == (20,)

    def test_fn_cate_bad_shape_raises(self):
        """fn_cate returning (n, K) with K > 1 raises ValueError."""
        def fn_fit(X, T, Y, **kw): return None
        def fn_cate(state, X): return np.ones((len(X), 3))  # wrong shape
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(*_dgp(n=50))
        with pytest.raises(ValueError, match="shape"):
            a.cate(np.zeros((20, 3)))

    def test_ate_default_is_mean_cate(self):
        """Without fn_ate, ate() = mean(cate(X_train))."""
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=100)
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(X, T, Y)
        ate_result = a.ate()
        cate_result = a.cate(X)
        assert ate_result.ate == pytest.approx(cate_result.cate.mean())

    def test_ate_x_none_uses_training_data(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=100)
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(X, T, Y)
        ate_none = a.ate(None)
        ate_x = a.ate(X)
        assert ate_none.ate == pytest.approx(ate_x.ate)

    def test_ate_with_new_x(self):
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=100)
        X_new = np.random.default_rng(7).normal(size=(50, 3))
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(X, T, Y)
        ate_train = a.ate()
        ate_new = a.ate(X_new)
        # Different X should generally give different ATE
        assert isinstance(ate_new, ComponentAteEstimate)


# ---------------------------------------------------------------------------
# GenericCATEAdapter — fn_ate (optional custom ATE)
# ---------------------------------------------------------------------------


class TestGenericCATEAdapterCustomATE:
    def test_fn_ate_called_when_provided(self):
        """fn_ate is used instead of mean(cate) when provided."""
        call_log = []

        def fn_fit(X, T, Y, **kw): return None
        def fn_cate(state, X): return np.zeros(len(X))
        def fn_ate(state, X):
            call_log.append(1)
            return 99.0  # distinctive sentinel

        a = GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        a.fit(*_dgp(n=50))
        result = a.ate()
        assert len(call_log) == 1
        assert result.ate == pytest.approx(99.0)

    def test_fn_ate_return_float(self):
        fn_fit, fn_cate, fn_ate = _tlearner_fns()
        X, T, Y = _dgp(n=100)
        a = GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        a.fit(X, T, Y)
        result = a.ate(X)
        assert isinstance(result, ComponentAteEstimate)
        assert np.isfinite(result.ate)

    def test_fn_ate_return_component_ate_estimate(self):
        """fn_ate can return a ComponentAteEstimate directly (e.g. with CIs)."""
        def fn_fit(X, T, Y, **kw): return None
        def fn_cate(state, X): return np.zeros(len(X))
        def fn_ate(state, X):
            return ComponentAteEstimate(ate=5.0)

        a = GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        a.fit(*_dgp(n=50))
        result = a.ate()
        assert result.ate == pytest.approx(5.0)

    def test_fn_ate_differs_from_mean_cate(self):
        """Custom fn_ate produces a different result than the default mean(cate)."""
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=200, seed=42)

        # Custom ATE: halve the result
        def fn_ate(state, X):
            m1, m0 = state
            return float((m1.predict(X) - m0.predict(X)).mean()) * 0.5

        a_default = GenericCATEAdapter(fn_fit, fn_cate)
        a_custom = GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        a_default.fit(X, T, Y)
        a_custom.fit(X, T, Y)

        ate_default = a_default.ate(X).ate
        ate_custom = a_custom.ate(X).ate
        assert ate_custom == pytest.approx(ate_default * 0.5, rel=1e-5)

    def test_fn_ate_receives_state_and_x(self):
        """Verify fn_ate receives the correct (state, X) arguments."""
        received = {}

        def fn_fit(X, T, Y, **kw): return {"marker": 42}
        def fn_cate(state, X): return np.zeros(len(X))
        def fn_ate(state, X):
            received["state"] = state
            received["n"] = len(X)
            return 0.0

        X, T, Y = _dgp(n=60)
        a = GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        a.fit(X, T, Y)
        a.ate(X)
        assert received["state"] == {"marker": 42}
        assert received["n"] == 60


# ---------------------------------------------------------------------------
# Deep-copy (required for bootstrap and cross-fitting)
# ---------------------------------------------------------------------------


class TestDeepCopy:
    def test_unfitted_adapter_is_deepcopyable(self):
        fn_fit, fn_cate, fn_ate = _tlearner_fns()
        a = GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        a_copy = copy.deepcopy(a)
        assert not a_copy._is_fitted

    def test_fitted_adapter_is_deepcopyable(self):
        fn_fit, fn_cate, fn_ate = _tlearner_fns()
        X, T, Y = _dgp()
        a = GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        a.fit(X, T, Y)
        a_copy = copy.deepcopy(a)
        # Copy should produce identical predictions
        np.testing.assert_array_equal(
            a.cate(X).cate, a_copy.cate(X).cate
        )

    def test_deepcopy_is_independent(self):
        """Mutating the copy's state doesn't affect the original."""
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp()
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(X, T, Y)
        a_copy = copy.deepcopy(a)
        a_copy._state = None  # corrupt the copy's state
        assert a._state is not None  # original unaffected


# ---------------------------------------------------------------------------
# random_state forwarding
# ---------------------------------------------------------------------------


class TestRandomStateForwarding:
    def test_random_state_forwarded_to_fn_fit(self):
        received = {}

        def fn_fit(X, T, Y, **kwargs):
            received["random_state"] = kwargs.get("random_state")
            return None

        def fn_cate(state, X): return np.zeros(len(X))

        X, T, Y = _dgp()
        a = GenericCATEAdapter(fn_fit, fn_cate)
        a.fit(X, T, Y, random_state=42)
        assert received["random_state"] == 42


# ---------------------------------------------------------------------------
# Integration with CausalEnsemble
# ---------------------------------------------------------------------------


class TestCausalEnsembleIntegration:
    def test_generic_cate_adapter_participates_in_cate(self):
        """GenericCATEAdapter contributes to ensemble CATE (not excluded)."""
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=150)
        ens = CausalEnsemble(
            methods=[GenericCATEAdapter(fn_fit, fn_cate, name="tlearner")],
            aggregation="median",
        )
        ens.fit(X, T, Y)
        result = ens.cate(X)
        assert result.cate.shape == (150,)
        assert "tlearner" in result.component_estimates

    def test_generic_cate_adapter_with_cba(self):
        """Works with agreement-based aggregation."""
        from metacausal.aggregation import CBA
        fn_fit, fn_cate, _ = _tlearner_fns()
        X, T, Y = _dgp(n=150)
        ens = CausalEnsemble(
            methods=[
                GenericCATEAdapter(fn_fit, fn_cate, name="a"),
                GenericCATEAdapter(fn_fit, fn_cate, name="b"),
            ],
            aggregation=CBA(),
        )
        ens.fit(X, T, Y)
        result = ens.cate(X)
        assert result.ensemble_weights is not None

    def test_generic_cate_with_custom_ate_in_ensemble(self):
        """fn_ate is used for component ATE when using pointwise strategy."""
        fn_fit, fn_cate, fn_ate = _tlearner_fns()
        X, T, Y = _dgp(n=150)
        ens = CausalEnsemble(
            methods=[GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate, name="t")],
            aggregation="median",
        )
        ens.fit(X, T, Y)
        ate_result = ens.ate(X)
        assert np.isfinite(ate_result.ate)

    def test_auto_wrapped_callable_still_uses_generic_ate_adapter(self):
        """CausalEnsemble auto-wraps a callable as GenericATEAdapter (not CATE)."""
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[lambda X, T, Y: 3.0],
            aggregation="median",
        )
        ens.fit(X, T, Y)
        adapter = ens._fitted_adapters[0]
        assert isinstance(adapter, GenericATEAdapter)
        assert not adapter.supports_cate

    def test_importable_from_top_level(self):
        from metacausal import GenericATEAdapter, GenericCATEAdapter  # noqa: F401
