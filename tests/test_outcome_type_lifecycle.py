"""Tests for CausalEnsemble outcome-type lifecycle.

Covers detection at fit time, the ``outcome_type`` override, deferred
default materialization, capability filtering, and per-component
validation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from metacausal import CausalEnsemble, ComponentExclusionWarning
from metacausal.adapters.generic import GenericATEAdapter
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _continuous_dgp(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(int)
    Y = X[:, 0] + 0.5 * T + rng.normal(scale=0.5, size=n)
    return X, T, Y


def _binary_dgp(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(int)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] + 0.7 * T)))
    Y = rng.binomial(1, p)
    return X, T, Y


class _DeclaredAdapter:
    """ATE-only adapter that declares its supported outcome types."""

    def __init__(self, name, ate_value, supported, outcome_validator=None):
        self._name = name
        self._ate = float(ate_value)
        self.supported_outcome_types = tuple(supported)
        self._validator = outcome_validator

    @property
    def name(self):
        return self._name

    @property
    def supports_cate(self):
        return False

    def validate_outcome_type(self, detected):
        if self._validator is not None:
            self._validator(detected)

    def fit(self, X, T, Y, **kwargs):
        pass

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._ate)

    def cate(self, X):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Outcome-type override validation at __init__
# ---------------------------------------------------------------------------


class TestOutcomeTypeArg:
    def test_default_is_auto(self):
        ens = CausalEnsemble(methods=[_DeclaredAdapter("a", 1.0, ("continuous",))])
        assert ens._outcome_type_request == "auto"

    @pytest.mark.parametrize("value", ["auto", "continuous", "binary"])
    def test_valid_values_accepted(self, value):
        ens = CausalEnsemble(
            methods=[_DeclaredAdapter("a", 1.0, ("continuous", "binary"))],
            outcome_type=value,
        )
        assert ens._outcome_type_request == value

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="outcome_type"):
            CausalEnsemble(
                methods=[_DeclaredAdapter("a", 1.0, ("continuous",))],
                outcome_type="survival",
            )


# ---------------------------------------------------------------------------
# Detection at fit time
# ---------------------------------------------------------------------------


class TestDetectionAtFit:
    def test_outcome_type_unset_before_fit(self):
        ens = CausalEnsemble(methods=[_DeclaredAdapter("a", 1.0, ("continuous",))])
        assert ens._outcome_type is None

    def test_continuous_detection(self):
        ens = CausalEnsemble(methods=[_DeclaredAdapter("a", 1.0, ("continuous",))])
        X, T, Y = _continuous_dgp()
        ens.fit(X, T, Y)
        assert ens._outcome_type == "continuous"

    def test_binary_detection(self):
        ens = CausalEnsemble(
            methods=[_DeclaredAdapter("a", 0.1, ("continuous", "binary"))]
        )
        X, T, Y = _binary_dgp()
        ens.fit(X, T, Y)
        assert ens._outcome_type == "binary"

    def test_explicit_binary_with_binary_y(self):
        ens = CausalEnsemble(
            methods=[_DeclaredAdapter("a", 0.1, ("continuous", "binary"))],
            outcome_type="binary",
        )
        X, T, Y = _binary_dgp()
        ens.fit(X, T, Y)
        assert ens._outcome_type == "binary"

    def test_explicit_binary_with_continuous_y_raises(self):
        ens = CausalEnsemble(
            methods=[_DeclaredAdapter("a", 1.0, ("continuous", "binary"))],
            outcome_type="binary",
        )
        X, T, Y = _continuous_dgp()
        with pytest.raises(ValueError, match="binary"):
            ens.fit(X, T, Y)

    def test_explicit_continuous_with_binary_y_silent(self):
        # Legitimate "give me the linear-probability ATE" request.
        ens = CausalEnsemble(
            methods=[_DeclaredAdapter("a", 0.1, ("continuous",))],
            outcome_type="continuous",
        )
        X, T, Y = _binary_dgp()
        ens.fit(X, T, Y)
        assert ens._outcome_type == "continuous"


# ---------------------------------------------------------------------------
# Capability filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_drops_continuous_only_components_on_binary(self):
        cont_only = _DeclaredAdapter("cont", 1.0, ("continuous",))
        both = _DeclaredAdapter("both", 0.2, ("continuous", "binary"))
        ens = CausalEnsemble(methods=[cont_only, both])
        X, T, Y = _binary_dgp()
        with pytest.warns(ComponentExclusionWarning, match="cont"):
            ens.fit(X, T, Y)
        assert [m.name for m in ens._fitted_adapters] == ["both"]

    def test_single_warning_for_multiple_drops(self):
        a = _DeclaredAdapter("a", 1.0, ("continuous",))
        b = _DeclaredAdapter("b", 1.0, ("continuous",))
        c = _DeclaredAdapter("c", 0.2, ("continuous", "binary"))
        ens = CausalEnsemble(methods=[a, b, c])
        X, T, Y = _binary_dgp()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ens.fit(X, T, Y)
        runtime = [w for w in caught if issubclass(w.category, ComponentExclusionWarning)
                   and "outcome type" in str(w.message).lower()]
        assert len(runtime) == 1
        assert "a" in str(runtime[0].message)
        assert "b" in str(runtime[0].message)

    def test_no_survivors_raises(self):
        a = _DeclaredAdapter("a", 1.0, ("continuous",))
        b = _DeclaredAdapter("b", 1.0, ("continuous",))
        ens = CausalEnsemble(methods=[a, b])
        X, T, Y = _binary_dgp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="No component supports"):
                ens.fit(X, T, Y)

    def test_no_warning_when_all_components_match(self):
        a = _DeclaredAdapter("a", 1.0, ("continuous",))
        b = _DeclaredAdapter("b", 1.0, ("continuous",))
        ens = CausalEnsemble(methods=[a, b])
        X, T, Y = _continuous_dgp()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ens.fit(X, T, Y)
        for w in caught:
            assert "outcome type" not in str(w.message).lower()


# ---------------------------------------------------------------------------
# Per-component validation (configuration check)
# ---------------------------------------------------------------------------


class TestComponentValidation:
    def test_validate_called_on_survivors(self):
        called = []

        def validator(detected):
            called.append(detected)

        a = _DeclaredAdapter(
            "a", 1.0, ("continuous", "binary"), outcome_validator=validator
        )
        ens = CausalEnsemble(methods=[a])
        X, T, Y = _continuous_dgp()
        ens.fit(X, T, Y)
        assert called == ["continuous"]

    def test_validation_failure_raises(self):
        def bad(detected):
            raise ValueError("misconfigured nuisance")

        a = _DeclaredAdapter(
            "a", 1.0, ("continuous",), outcome_validator=bad
        )
        ens = CausalEnsemble(methods=[a])
        X, T, Y = _continuous_dgp()
        with pytest.raises(ValueError, match="misconfigured"):
            ens.fit(X, T, Y)


# ---------------------------------------------------------------------------
# Deferred default materialization
# ---------------------------------------------------------------------------


class TestDeferredDefaults:
    def test_method_names_empty_pre_fit(self):
        ens = CausalEnsemble()  # default methods, deferred
        assert ens.method_names == []

    def test_default_pool_materialized_after_continuous_fit(self):
        ens = CausalEnsemble()
        X, T, Y = _continuous_dgp(n=100)
        # Disable any internal warnings from the slow fits at runtime.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ens.fit(X, T, Y, random_state=0)
        # Continuous default pool has 10 components.
        assert len(ens._fitted_adapters) >= 8  # some may fail on tiny n
        assert ens._outcome_type == "continuous"

    def test_default_pool_materialized_after_binary_fit(self):
        ens = CausalEnsemble()
        X, T, Y = _binary_dgp(n=400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ens.fit(X, T, Y, random_state=0)
        # Binary default pool has 8 components.
        names = {m.name for m in ens._fitted_adapters}
        assert "BCF" not in names  # stochtree dropped
        assert "DoubleMLPLR" not in names  # PLR dropped
        assert "DoubleMLIRM" in names
        assert ens._outcome_type == "binary"


# ---------------------------------------------------------------------------
# Refit re-runs detection and filtering
# ---------------------------------------------------------------------------


class TestRefit:
    def test_user_supplied_pool_re_filters_on_refit(self):
        cont_only = _DeclaredAdapter("cont", 1.0, ("continuous",))
        both = _DeclaredAdapter("both", 0.2, ("continuous", "binary"))
        ens = CausalEnsemble(methods=[cont_only, both])

        X1, T1, Y1 = _continuous_dgp(n=100, seed=1)
        ens.fit(X1, T1, Y1)
        assert {m.name for m in ens._fitted_adapters} == {"cont", "both"}

        # Refit on binary Y — should drop cont_only via filtering.
        X2, T2, Y2 = _binary_dgp(n=100, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ens.fit(X2, T2, Y2)
        assert {m.name for m in ens._fitted_adapters} == {"both"}


# ---------------------------------------------------------------------------
# Generic adapter binary opt-in
# ---------------------------------------------------------------------------


class TestGenericBinaryOptIn:
    def test_default_generic_dropped_on_binary(self):
        adapter = GenericATEAdapter(lambda X, T, Y: 0.0, name="g")
        ens = CausalEnsemble(methods=[adapter])
        X, T, Y = _binary_dgp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="No component"):
                ens.fit(X, T, Y)

    def test_opted_in_generic_kept_on_binary(self):
        adapter = GenericATEAdapter(
            lambda X, T, Y: 0.0,
            name="g",
            supported_outcome_types=("continuous", "binary"),
        )
        ens = CausalEnsemble(methods=[adapter])
        X, T, Y = _binary_dgp()
        ens.fit(X, T, Y)
        assert [m.name for m in ens._fitted_adapters] == ["g"]


# ---------------------------------------------------------------------------
# Rejected outcome inputs to fit()
# ---------------------------------------------------------------------------


class TestRejectedOutcomes:
    def test_categorical_string_y_raises(self):
        ens = CausalEnsemble(methods=[_DeclaredAdapter("a", 1.0, ("continuous",))])
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        T = rng.binomial(1, 0.5, size=50)
        Y = np.array(["yes", "no"] * 25)
        with pytest.raises(ValueError, match="non-numeric"):
            ens.fit(X, T, Y)

    def test_nan_y_raises(self):
        ens = CausalEnsemble(methods=[_DeclaredAdapter("a", 1.0, ("continuous",))])
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        T = rng.binomial(1, 0.5, size=50)
        Y = rng.normal(size=50)
        Y[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ens.fit(X, T, Y)


# ---------------------------------------------------------------------------
# Real-adapter integration: filter and validate paths
# ---------------------------------------------------------------------------


class TestRealAdapterIntegration:
    def test_econml_slearner_dropped_on_binary(self):
        """SLearner declares continuous-only; should be filtered out."""
        from econml.metalearners import SLearner
        from sklearn.ensemble import HistGradientBoostingRegressor

        from metacausal.adapters.causalml import CausalMLAdapter
        from causalml.inference.meta import BaseSClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier

        slearner = SLearner(overall_model=HistGradientBoostingRegressor())
        binary_capable = BaseSClassifier(learner=HistGradientBoostingClassifier())

        ens = CausalEnsemble(methods=[slearner, binary_capable])
        X, T, Y = _binary_dgp(n=300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ens.fit(X, T, Y, random_state=0)

        runtime = [
            w for w in caught
            if issubclass(w.category, ComponentExclusionWarning)
            and "outcome type" in str(w.message).lower()
        ]
        assert len(runtime) == 1
        assert "SLearner" in str(runtime[0].message)
        names = {m.name for m in ens._fitted_adapters}
        assert names == {"BaseSClassifier"}

    def test_doubleml_irm_with_regressor_ml_g_raises_on_binary(self):
        """DoubleMLIRM supports binary, but the user wired a regressor
        into ml_g — validate_outcome_type raises before fit begins."""
        from doubleml import DoubleMLIRM
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        from metacausal.adapters.doubleml import DoubleMLAdapter

        adapter = DoubleMLAdapter(
            DoubleMLIRM,
            ml_g=HistGradientBoostingRegressor(),  # wrong: should be classifier for binary Y
            ml_m=HistGradientBoostingClassifier(),
        )
        ens = CausalEnsemble(methods=[adapter])
        X, T, Y = _binary_dgp(n=200)
        with pytest.raises(ValueError, match="classifier"):
            ens.fit(X, T, Y)
