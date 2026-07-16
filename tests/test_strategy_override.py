"""Tests for the aggregation override parameter on cate() and ate()."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal import CausalEnsemble
from metacausal.aggregation import (
    CBA,
    CrossFitSplit,
    Mean,
    Median,
    SupervisedStrategy,
    TrainAvgSplit,
)
from metacausal.aggregation.weights import EnsembleWeights

# Reuse helpers from test_supervised_fit
from tests.test_supervised_fit import (
    CateCapableAdapter,
    UniformSupervisedStrategy,
    _dgp,
)
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


class VaryingCateAdapter:
    """Adapter whose CATE predictions vary with X (non-constant), suitable for CBA."""

    def __init__(self, name: str, coef: float = 1.0):
        self._name = name
        self._coef = coef

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, random_state=None, **kwargs):
        pass

    def ate(self, X=None):
        return ComponentAteEstimate(ate=float(self._coef))

    def cate(self, X):
        return ComponentCateEstimate(cate=self._coef * X[:, 0])

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_supervised_strategy(n_folds: int = 3) -> UniformSupervisedStrategy:
    return UniformSupervisedStrategy(
        split=CrossFitSplit(n_folds=n_folds, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
    )


def _fitted_ensemble(n: int = 200, aggregation=None, seed: int = 0):
    """Return a fitted ensemble with two CateCapableAdapters."""
    X, T, Y = _dgp(n=n, seed=seed)
    if aggregation is None:
        aggregation = "median"
    ens = CausalEnsemble(
        methods=[CateCapableAdapter("a", scale=1.0), CateCapableAdapter("b", scale=3.0)],
        aggregation=aggregation,
    )
    ens.fit(X, T, Y, random_state=seed)
    return ens, X, T, Y


# ---------------------------------------------------------------------------
# aggregation=None uses default strategy
# ---------------------------------------------------------------------------


class TestNoneOverride:
    def test_cate_none_uses_default(self):
        ens, X, *_ = _fitted_ensemble()
        result_default = ens.cate(X)
        result_none = ens.cate(X, aggregation=None)
        np.testing.assert_array_equal(result_default.cate, result_none.cate)

    def test_ate_none_uses_default(self):
        ens, X, *_ = _fitted_ensemble()
        assert ens.ate(X).ate == ens.ate(X, aggregation=None).ate

    def test_aggregation_field_reflects_default(self):
        ens, X, *_ = _fitted_ensemble(aggregation="median")
        result = ens.cate(X, aggregation=None)
        assert result.aggregation == "Median"


# ---------------------------------------------------------------------------
# Pointwise overrides
# ---------------------------------------------------------------------------


class TestPointwiseOverride:
    def test_string_alias_median(self):
        ens, X, *_ = _fitted_ensemble(aggregation="mean")
        result = ens.cate(X, aggregation="median")
        assert result.aggregation == "Median"

    def test_string_alias_mean(self):
        ens, X, *_ = _fitted_ensemble(aggregation="median")
        result = ens.cate(X, aggregation="mean")
        assert result.aggregation == "Mean"
        np.testing.assert_allclose(result.cate, 2.0)  # (1+3)/2

    def test_strategy_object_median(self):
        ens, X, *_ = _fitted_ensemble(aggregation="mean")
        result = ens.cate(X, aggregation=Median())
        assert result.aggregation == "Median"

    def test_strategy_object_mean(self):
        ens, X, *_ = _fitted_ensemble(aggregation="median")
        result = ens.cate(X, aggregation=Mean())
        np.testing.assert_allclose(result.cate, 2.0)

    def test_ensemble_weights_none_for_pointwise_override(self):
        ens, X, *_ = _fitted_ensemble(aggregation="mean")
        result = ens.cate(X, aggregation=Median())
        assert result.ensemble_weights is None

    def test_ate_with_string_override(self):
        ens, X, *_ = _fitted_ensemble(aggregation="mean")
        result = ens.ate(X, aggregation="median")
        assert result.aggregation == "Median"

    def test_invalid_string_raises(self):
        ens, X, *_ = _fitted_ensemble()
        with pytest.raises((ValueError, KeyError)):
            ens.cate(X, aggregation="nonexistent")


# ---------------------------------------------------------------------------
# AgreementStrategy (CBA) override
# ---------------------------------------------------------------------------


def _varying_ensemble(n: int = 200, aggregation=None, seed: int = 0):
    """Return a fitted ensemble with non-constant CATE adapters (required for CBA)."""
    X, T, Y = _dgp(n=n, seed=seed)
    if aggregation is None:
        aggregation = "median"
    ens = CausalEnsemble(
        methods=[VaryingCateAdapter("a", coef=1.0), VaryingCateAdapter("b", coef=2.0)],
        aggregation=aggregation,
    )
    ens.fit(X, T, Y, random_state=seed)
    return ens, X, T, Y


class TestAgreementOverride:
    def test_cba_override_after_pointwise_fit(self):
        ens, X, *_ = _varying_ensemble(aggregation="median")
        result = ens.cate(X, aggregation=CBA())
        assert result.aggregation == "CBA"
        assert result.ensemble_weights is not None

    def test_cba_override_after_supervised_fit(self):
        ens, X, *_ = _varying_ensemble(aggregation=_make_supervised_strategy())
        result = ens.cate(X, aggregation=CBA())
        assert result.aggregation == "CBA"
        assert result.ensemble_weights is not None

    def test_cba_weights_sum_to_one(self):
        ens, X, *_ = _varying_ensemble(aggregation="median")
        result = ens.cate(X, aggregation=CBA())
        np.testing.assert_allclose(result.ensemble_weights.weights.sum(), 1.0)

    def test_cba_override_independent_calls(self):
        """Two CBA() instances produce the same weights (deterministic)."""
        ens, X, *_ = _varying_ensemble(aggregation="median")
        r1 = ens.cate(X, aggregation=CBA())
        r2 = ens.cate(X, aggregation=CBA())
        np.testing.assert_array_equal(
            r1.ensemble_weights.weights, r2.ensemble_weights.weights
        )

    def test_ate_with_cba_override(self):
        ens, X, *_ = _varying_ensemble(aggregation="median")
        ate_result = ens.ate(X, aggregation=CBA())
        cate_result = ens.cate(X, aggregation=CBA())
        np.testing.assert_allclose(ate_result.ate, cate_result.cate.mean())


# ---------------------------------------------------------------------------
# SupervisedStrategy override
# ---------------------------------------------------------------------------


class TestSupervisedOverride:
    def test_supervised_override_after_supervised_fit(self):
        ens, X, *_ = _fitted_ensemble(aggregation=_make_supervised_strategy())
        override = _make_supervised_strategy()
        result = ens.cate(X, aggregation=override)
        assert result.aggregation == "UniformSupervisedStrategy"
        assert result.ensemble_weights is not None

    def test_supervised_override_uses_cached_oof(self):
        """Override uses same OOF predictions as original fit."""
        ens, X, *_ = _fitted_ensemble(aggregation=_make_supervised_strategy())
        original_oof = ens._cached_oof_cate_predictions.copy()

        override = _make_supervised_strategy()
        ens.cate(X, aggregation=override)

        # The OOF cache is unchanged after an override call
        np.testing.assert_array_equal(ens._cached_oof_cate_predictions, original_oof)

    def test_supervised_override_model_names_set(self):
        ens, X, *_ = _fitted_ensemble(aggregation=_make_supervised_strategy())
        override = _make_supervised_strategy()
        result = ens.cate(X, aggregation=override)
        assert set(result.ensemble_weights.model_names) == {"a", "b"}

    def test_supervised_override_after_pointwise_fit_raises(self):
        ens, X, *_ = _fitted_ensemble(aggregation="median")
        with pytest.raises(RuntimeError, match="No cross-fitting artifacts"):
            ens.cate(X, aggregation=_make_supervised_strategy())

    def test_supervised_override_after_agreement_fit_raises(self):
        ens, X, *_ = _varying_ensemble(aggregation=CBA())
        with pytest.raises(RuntimeError, match="No cross-fitting artifacts"):
            ens.cate(X, aggregation=_make_supervised_strategy())

    def test_ate_with_supervised_override(self):
        ens, X, *_ = _fitted_ensemble(aggregation=_make_supervised_strategy())
        override = _make_supervised_strategy()
        ate_result = ens.ate(X, aggregation=override)
        # ATE = mean(CATE) for supervised strategy
        override2 = _make_supervised_strategy()
        cate_result = ens.cate(X, aggregation=override2)
        np.testing.assert_allclose(ate_result.ate, cate_result.cate.mean())

    def test_supervised_override_with_train_avg_split_fit(self):
        """Override works when original fit used TrainAvgSplit."""
        X, T, Y = _dgp(n=200)
        strategy = UniformSupervisedStrategy(
            split=TrainAvgSplit(avg_frac=0.25, stratify=False),
            propensity_model=LogisticRegression(max_iter=200),
            outcome_model=LinearRegression(),
        )
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=strategy,
        )
        ens.fit(X, T, Y, random_state=0)

        override = _make_supervised_strategy()
        result = ens.cate(X, aggregation=override)
        assert result.ensemble_weights is not None


# ---------------------------------------------------------------------------
# Config mismatch warning
# ---------------------------------------------------------------------------


def _default_nuisance_supervised_ensemble(n: int = 200, seed: int = 0):
    """Ensemble fitted with a supervised strategy using default nuisance (None models)."""
    X, T, Y = _dgp(n=n, seed=seed)
    strategy = UniformSupervisedStrategy(
        split=CrossFitSplit(n_folds=3, stratify=False),
        propensity_model=None,   # uses default LightGBM
        outcome_model=None,
    )
    ens = CausalEnsemble(
        methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
        aggregation=strategy,
    )
    ens.fit(X, T, Y, random_state=seed)
    return ens, X


class TestConfigMismatchWarning:
    def test_no_warning_when_configs_match(self):
        """No warning when override has same config as original (both default None models)."""
        ens, X = _default_nuisance_supervised_ensemble()
        override = UniformSupervisedStrategy(
            split=CrossFitSplit(n_folds=3, stratify=False),
            propensity_model=None,
            outcome_model=None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ens.cate(X, aggregation=override)
        mismatch_warns = [w for w in caught if "ignored" in str(w.message).lower()]
        assert len(mismatch_warns) == 0

    def test_warning_when_propensity_model_added(self):
        """Warn when override adds a propensity_model where original used default (None)."""
        ens, X = _default_nuisance_supervised_ensemble()
        override = UniformSupervisedStrategy(
            split=CrossFitSplit(n_folds=3, stratify=False),
            propensity_model=LogisticRegression(max_iter=100),  # non-None on None original
            outcome_model=None,
        )
        with pytest.warns(UserWarning, match="ignored"):
            ens.cate(X, aggregation=override)

    def test_warning_when_outcome_model_added(self):
        """Warn when override adds an outcome_model where original used default (None)."""
        ens, X = _default_nuisance_supervised_ensemble()
        override = UniformSupervisedStrategy(
            split=CrossFitSplit(n_folds=3, stratify=False),
            propensity_model=None,
            outcome_model=LinearRegression(),  # non-None on None original
        )
        with pytest.warns(UserWarning, match="ignored"):
            ens.cate(X, aggregation=override)

    def test_warning_when_split_type_differs(self):
        """Warn when override uses a different split type than original."""
        ens, X = _default_nuisance_supervised_ensemble()
        override = UniformSupervisedStrategy(
            split=TrainAvgSplit(avg_frac=0.25),  # different split type
            propensity_model=None,
            outcome_model=None,
        )
        with pytest.warns(UserWarning, match="ignored"):
            ens.cate(X, aggregation=override)

    def test_no_warning_for_supervised_override_after_pointwise_fit(self):
        """No config warning (RuntimeError instead) when OOF cache missing."""
        ens, X, *_ = _fitted_ensemble(aggregation="median")
        with pytest.raises(RuntimeError, match="No cross-fitting artifacts"):
            ens.cate(X, aggregation=_make_supervised_strategy())
