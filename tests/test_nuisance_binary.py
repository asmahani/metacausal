"""Binary-Y path through fit_nuisance and supervised aggregation.

Verifies that the cross-fitted nuisance machinery handles binary outcomes
correctly: classifier outcome models, ``predict_proba``-based mu_hat,
and the doubly-robust pseudo-outcome targeting the risk-difference ATE.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression

from metacausal import CausalEnsemble
from metacausal.aggregation import CrossFitSplit
from metacausal.aggregation.causal_stacking import CausalStacking
from metacausal.aggregation.nuisance import (
    NuisanceEstimates,
    dr_pseudo_outcome,
    fit_nuisance,
)
from metacausal.aggregation.q_aggregation import QAggregation
from metacausal.aggregation.r_stacking import RStacking
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_dgp(n=400, seed=0):
    """Binary outcome with logistic structural form. True RD ≈ 0.16."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-0.3 * X[:, 0]))
    T = rng.binomial(1, e).astype(float)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] - 1.0 + 0.7 * T)))
    Y = rng.binomial(1, p)
    return X, T, Y


class _ConstantCateAdapter:
    """CATE-capable adapter returning a constant CATE for testing."""

    def __init__(self, name, value):
        self._name = name
        self._value = float(value)

    @property
    def name(self):
        return self._name

    @property
    def supports_cate(self):
        return True

    supported_outcome_types = ("continuous", "binary")

    def validate_outcome_type(self, detected):
        return

    def fit(self, X, T, Y, **kwargs):
        pass

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._value)

    def cate(self, X):
        return ComponentCateEstimate(cate=np.full(X.shape[0], self._value))


# ---------------------------------------------------------------------------
# fit_nuisance binary path
# ---------------------------------------------------------------------------


class TestFitNuisanceBinary:
    def test_mu_hat_in_unit_interval(self):
        X, T, Y = _binary_dgp(n=500)
        fold_spec = CrossFitSplit(n_folds=3, stratify=False).split(T)
        nuisance = fit_nuisance(
            X, T, Y, fold_spec,
            propensity_model=LogisticRegression(max_iter=500),
            outcome_model=HistGradientBoostingClassifier(max_iter=200),
            random_state=0,
            outcome_type="binary",
        )
        assert np.all(nuisance.mu1_hat >= 0)
        assert np.all(nuisance.mu1_hat <= 1)
        assert np.all(nuisance.mu0_hat >= 0)
        assert np.all(nuisance.mu0_hat <= 1)

    def test_auto_detection_of_outcome_type(self):
        X, T, Y = _binary_dgp(n=500)
        fold_spec = CrossFitSplit(n_folds=3, stratify=False).split(T)
        # Don't pass outcome_type; detection should pick "binary" from Y.
        nuisance = fit_nuisance(
            X, T, Y, fold_spec,
            propensity_model=LogisticRegression(max_iter=500),
            outcome_model=HistGradientBoostingClassifier(max_iter=200),
            random_state=0,
        )
        assert np.all(nuisance.mu1_hat >= 0)
        assert np.all(nuisance.mu1_hat <= 1)

    def test_default_outcome_model_picks_classifier_for_binary(self):
        X, T, Y = _binary_dgp(n=500)
        fold_spec = CrossFitSplit(n_folds=3, stratify=False).split(T)
        # No outcome_model supplied; should pick HistGradientBoostingClassifier.
        nuisance = fit_nuisance(
            X, T, Y, fold_spec,
            random_state=0,
            outcome_type="binary",
        )
        assert np.all(nuisance.mu1_hat >= 0) and np.all(nuisance.mu1_hat <= 1)

    def test_dr_pseudo_outcome_recovers_risk_difference(self):
        X, T, Y = _binary_dgp(n=2000, seed=1)
        true_rd = float(np.mean(
            1.0 / (1.0 + np.exp(-(X[:, 0] - 1.0 + 0.7)))
            - 1.0 / (1.0 + np.exp(-(X[:, 0] - 1.0)))
        ))
        fold_spec = CrossFitSplit(n_folds=5, stratify=False).split(T)
        nuisance = fit_nuisance(
            X, T, Y, fold_spec,
            propensity_model=LogisticRegression(max_iter=500),
            outcome_model=HistGradientBoostingClassifier(max_iter=200),
            random_state=0,
            outcome_type="binary",
        )
        gamma = dr_pseudo_outcome(Y, T, nuisance)
        ate_dr = float(np.mean(gamma))
        assert abs(ate_dr - true_rd) < 0.05, (
            f"DR ATE {ate_dr:.4f} vs true RD {true_rd:.4f}"
        )

    def test_regressor_with_binary_y_raises(self):
        X, T, Y = _binary_dgp(n=200)
        fold_spec = CrossFitSplit(n_folds=2, stratify=False).split(T)
        with pytest.raises(ValueError, match="predict_proba"):
            fit_nuisance(
                X, T, Y, fold_spec,
                outcome_model=HistGradientBoostingRegressor(),
                outcome_type="binary",
            )

    def test_classifier_with_continuous_y_raises(self):
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 3))
        T = rng.binomial(1, 0.5, size=n)
        Y = rng.normal(size=n)
        fold_spec = CrossFitSplit(n_folds=2, stratify=False).split(T)
        with pytest.raises(ValueError, match="classifier"):
            fit_nuisance(
                X, T, Y, fold_spec,
                outcome_model=HistGradientBoostingClassifier(),
                outcome_type="continuous",
            )

    def test_invalid_outcome_type_raises(self):
        X, T, Y = _binary_dgp(n=100)
        fold_spec = CrossFitSplit(n_folds=2, stratify=False).split(T)
        with pytest.raises(ValueError, match="outcome_type"):
            fit_nuisance(X, T, Y, fold_spec, outcome_type="survival")


# ---------------------------------------------------------------------------
# Supervised aggregators end-to-end on binary Y
# ---------------------------------------------------------------------------


class TestSupervisedAggregatorsBinary:
    @pytest.fixture
    def methods(self):
        # Two CATE-capable adapters whose constant CATEs straddle the
        # true RD; supervised stacking should weight them sensibly.
        return [
            _ConstantCateAdapter("low", 0.0),
            _ConstantCateAdapter("high", 0.3),
        ]

    @pytest.mark.parametrize(
        "strategy_factory",
        [
            lambda: CausalStacking(
                split=CrossFitSplit(n_folds=3, stratify=False),
            ),
            lambda: RStacking(
                split=CrossFitSplit(n_folds=3, stratify=False),
            ),
            lambda: QAggregation(
                split=CrossFitSplit(n_folds=3, stratify=False),
            ),
        ],
        ids=["CausalStacking", "RStacking", "QAggregation"],
    )
    def test_strategy_runs_on_binary_y(self, strategy_factory, methods):
        X, T, Y = _binary_dgp(n=500, seed=0)
        ens = CausalEnsemble(methods=methods, aggregation=strategy_factory())
        ens.fit(X, T, Y, random_state=0)

        result = ens.cate(X)
        assert np.all(np.isfinite(result.cate))
        # CATE estimates should be in a plausible RD range — not blown up.
        assert np.all(np.abs(result.cate) <= 1.0)
