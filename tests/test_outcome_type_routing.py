"""Tests for adapter outcome-type capability declaration and validation.

Verifies, for each adapter, that ``supported_outcome_types`` correctly
reflects what the wrapped class can handle and that
``validate_outcome_type`` raises on mismatched user configurations.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from metacausal.adapters.causalml import CausalMLAdapter
from metacausal.adapters.doubleml import DoubleMLAdapter
from metacausal.adapters.econml import EconMLAdapter
from metacausal.adapters.generic import GenericATEAdapter, GenericCATEAdapter
from metacausal.adapters.stochtree import StochtreeAdapter


# ---------------------------------------------------------------------------
# Fakes — mimic library classes by name so we can test capability
# declaration without importing the heavy underlying libraries.
# ---------------------------------------------------------------------------


def _make_fake(class_name: str, **attrs):
    """Build an instance whose ``type(...).__name__`` is *class_name*."""
    cls = type(class_name, (SimpleNamespace,), {})
    return cls(**attrs)


# ---------------------------------------------------------------------------
# StochtreeAdapter
# ---------------------------------------------------------------------------


class TestStochtreeCapability:
    def test_supports_continuous_only(self):
        adapter = StochtreeAdapter()
        assert adapter.supported_outcome_types == ("continuous",)

    def test_validate_is_noop(self):
        adapter = StochtreeAdapter()
        adapter.validate_outcome_type("continuous")
        adapter.validate_outcome_type("binary")  # also a no-op


# ---------------------------------------------------------------------------
# GenericATEAdapter / GenericCATEAdapter
# ---------------------------------------------------------------------------


class TestGenericCapability:
    def test_ate_default_continuous(self):
        adapter = GenericATEAdapter(lambda X, T, Y: 0.0, name="x")
        assert adapter.supported_outcome_types == ("continuous",)

    def test_ate_user_opt_in_binary(self):
        adapter = GenericATEAdapter(
            lambda X, T, Y: 0.0,
            name="x",
            supported_outcome_types=("continuous", "binary"),
        )
        assert adapter.supported_outcome_types == ("continuous", "binary")

    def test_ate_validate_is_noop(self):
        adapter = GenericATEAdapter(lambda X, T, Y: 0.0)
        adapter.validate_outcome_type("continuous")
        adapter.validate_outcome_type("binary")

    def test_cate_default_continuous(self):
        adapter = GenericCATEAdapter(
            fn_fit=lambda X, T, Y, **_: None,
            fn_cate=lambda state, X: np.zeros(X.shape[0]),
        )
        assert adapter.supported_outcome_types == ("continuous",)

    def test_cate_user_opt_in_binary(self):
        adapter = GenericCATEAdapter(
            fn_fit=lambda X, T, Y, **_: None,
            fn_cate=lambda state, X: np.zeros(X.shape[0]),
            supported_outcome_types=("continuous", "binary"),
        )
        assert adapter.supported_outcome_types == ("continuous", "binary")

    def test_cate_validate_is_noop(self):
        adapter = GenericCATEAdapter(
            fn_fit=lambda X, T, Y, **_: None,
            fn_cate=lambda state, X: np.zeros(X.shape[0]),
        )
        adapter.validate_outcome_type("continuous")
        adapter.validate_outcome_type("binary")


# ---------------------------------------------------------------------------
# DoubleMLAdapter
# ---------------------------------------------------------------------------


class TestDoubleMLCapability:
    def test_irm_supports_both(self):
        fake_class = type("DoubleMLIRM", (), {})
        adapter = DoubleMLAdapter(fake_class)
        assert adapter.supported_outcome_types == ("continuous", "binary")

    def test_plr_continuous_only(self):
        fake_class = type("DoubleMLPLR", (), {})
        adapter = DoubleMLAdapter(fake_class)
        assert adapter.supported_outcome_types == ("continuous",)

    def test_unknown_class_continuous_only(self):
        fake_class = type("DoubleMLAPO", (), {})
        adapter = DoubleMLAdapter(fake_class)
        assert adapter.supported_outcome_types == ("continuous",)


class TestDoubleMLValidation:
    def _adapter(self, **kwargs):
        fake_class = type("DoubleMLIRM", (), {})
        return DoubleMLAdapter(fake_class, **kwargs)

    def test_continuous_passes_with_regressor_ml_g(self):
        adapter = self._adapter(ml_g=HistGradientBoostingRegressor())
        adapter.validate_outcome_type("continuous")  # no raise

    def test_binary_passes_with_classifier_ml_g(self):
        adapter = self._adapter(ml_g=HistGradientBoostingClassifier())
        adapter.validate_outcome_type("binary")  # no raise

    def test_binary_raises_on_regressor_ml_g(self):
        adapter = self._adapter(ml_g=HistGradientBoostingRegressor())
        with pytest.raises(ValueError, match="classifier"):
            adapter.validate_outcome_type("binary")

    def test_binary_no_ml_g_is_silent(self):
        adapter = self._adapter()  # no ml_g supplied
        adapter.validate_outcome_type("binary")  # no raise


# ---------------------------------------------------------------------------
# EconMLAdapter
# ---------------------------------------------------------------------------


class TestEconMLCapability:
    def test_causal_forest_dml_supports_both(self):
        model = _make_fake("CausalForestDML", discrete_outcome=False, model_y=None)
        adapter = EconMLAdapter(model)
        assert adapter.supported_outcome_types == ("continuous", "binary")

    def test_dr_learner_supports_both(self):
        model = _make_fake(
            "DRLearner", discrete_outcome=False, model_regression=None
        )
        adapter = EconMLAdapter(model)
        assert adapter.supported_outcome_types == ("continuous", "binary")

    @pytest.mark.parametrize("cls_name", ["SLearner", "TLearner", "XLearner"])
    def test_metalearners_continuous_only(self, cls_name):
        model = _make_fake(cls_name)
        adapter = EconMLAdapter(model)
        assert adapter.supported_outcome_types == ("continuous",)


class TestEconMLValidation:
    def test_continuous_passes_regardless(self):
        model = _make_fake(
            "CausalForestDML", discrete_outcome=False, model_y=None
        )
        adapter = EconMLAdapter(model)
        adapter.validate_outcome_type("continuous")  # no raise

    def test_binary_requires_discrete_outcome_flag(self):
        model = _make_fake(
            "CausalForestDML",
            discrete_outcome=False,
            model_y=HistGradientBoostingClassifier(),
        )
        adapter = EconMLAdapter(model)
        with pytest.raises(ValueError, match="discrete_outcome"):
            adapter.validate_outcome_type("binary")

    def test_binary_requires_classifier_model_y(self):
        model = _make_fake(
            "CausalForestDML",
            discrete_outcome=True,
            model_y=HistGradientBoostingRegressor(),
        )
        adapter = EconMLAdapter(model)
        with pytest.raises(ValueError, match="classifier"):
            adapter.validate_outcome_type("binary")

    def test_binary_passes_with_classifier_model_y(self):
        model = _make_fake(
            "CausalForestDML",
            discrete_outcome=True,
            model_y=HistGradientBoostingClassifier(),
        )
        adapter = EconMLAdapter(model)
        adapter.validate_outcome_type("binary")  # no raise

    def test_binary_dr_learner_checks_model_regression(self):
        model = _make_fake(
            "DRLearner",
            discrete_outcome=True,
            model_regression=HistGradientBoostingRegressor(),
        )
        adapter = EconMLAdapter(model)
        with pytest.raises(ValueError, match="model_regression"):
            adapter.validate_outcome_type("binary")


# ---------------------------------------------------------------------------
# CausalMLAdapter
# ---------------------------------------------------------------------------


class TestCausalMLCapability:
    @pytest.mark.parametrize(
        "cls_name",
        [
            "BaseSRegressor",
            "BaseTRegressor",
            "BaseXRegressor",
            "BaseDRRegressor",
            "BaseRRegressor",
            "CausalTreeRegressor",
        ],
    )
    def test_regressors_continuous_only(self, cls_name):
        model = _make_fake(cls_name)
        adapter = CausalMLAdapter(model)
        assert adapter.supported_outcome_types == ("continuous",)

    @pytest.mark.parametrize(
        "cls_name",
        [
            "BaseSClassifier",
            "BaseTClassifier",
            "BaseXClassifier",
            "BaseDRClassifier",
            "BaseRClassifier",
            "UpliftTreeClassifier",
            "UpliftRandomForestClassifier",
        ],
    )
    def test_classifiers_binary_only(self, cls_name):
        model = _make_fake(cls_name)
        adapter = CausalMLAdapter(model)
        assert adapter.supported_outcome_types == ("binary",)

    def test_tmle_supports_both(self):
        model = _make_fake("TMLELearner", learner=HistGradientBoostingRegressor())
        adapter = CausalMLAdapter(model)
        assert adapter.supported_outcome_types == ("continuous", "binary")


class TestCausalMLValidation:
    def test_tmle_with_regressor_passes(self):
        model = _make_fake("TMLELearner", learner=HistGradientBoostingRegressor())
        adapter = CausalMLAdapter(model)
        adapter.validate_outcome_type("continuous")  # no raise
        adapter.validate_outcome_type("binary")  # no raise

    def test_tmle_with_classifier_raises(self):
        model = _make_fake("TMLELearner", learner=HistGradientBoostingClassifier())
        adapter = CausalMLAdapter(model)
        with pytest.raises(ValueError, match="regressor"):
            adapter.validate_outcome_type("binary")

    def test_non_tmle_validate_is_noop(self):
        # For meta-learner regressors/classifiers, capability declaration
        # handles routing; per-instance validation is a no-op.
        model = _make_fake("BaseRClassifier")
        adapter = CausalMLAdapter(model)
        adapter.validate_outcome_type("binary")  # no raise
        adapter.validate_outcome_type("continuous")  # no raise either
