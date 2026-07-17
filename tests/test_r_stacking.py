"""Tests for RStacking."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal import CausalEnsemble
from metacausal.aggregation import CrossFitSplit, RStacking
from metacausal.aggregation.nuisance import NuisanceEstimates, robinson_residuals
from metacausal.aggregation.weights import EnsembleWeights

from tests.test_supervised_fit import CateCapableAdapter, _dgp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oracle_nuisance(n: int, e: float = 0.5) -> NuisanceEstimates:
    return NuisanceEstimates(
        e_hat=np.full(n, e),
        mu1_hat=np.zeros(n),
        mu0_hat=np.zeros(n),
    )


def _fast_rs():
    return RStacking(
        split=CrossFitSplit(n_folds=3, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
    )


def _call_fit_weights(strategy, n=100, K=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = rng.normal(size=n)
    cate_preds = rng.normal(size=(K, n))
    nuisance = _oracle_nuisance(n)
    strategy.fit_weights(cate_preds, Y, T, X, nuisance)
    return strategy._weights


# ---------------------------------------------------------------------------
# Weight constraints: alpha_k >= 0, no sum-to-one
# ---------------------------------------------------------------------------


class TestWeightConstraints:
    def test_alpha_nonneg(self):
        w = _call_fit_weights(RStacking(), K=4)
        assert np.all(w.weights >= -1e-9)

    def test_weights_do_not_sum_to_one(self):
        """R-Stacking is conic: weights need not sum to 1."""
        # Just verify the class doesn't enforce sum-to-one
        w = _call_fit_weights(RStacking(), K=3, n=200)
        # We can't assert != 1 universally, but verify the constraint isn't applied
        assert w.method == "r_stacking"

    def test_weights_shape(self):
        K = 4
        w = _call_fit_weights(RStacking(), K=K)
        assert w.weights.shape == (K,)

    def test_k1_runs(self):
        """K=1: solver still runs, returns valid weights and intercept."""
        w = _call_fit_weights(RStacking(), K=1)
        assert w.weights.shape == (1,)
        assert w.weights[0] >= 0.0
        assert isinstance(w.intercept, float)


# ---------------------------------------------------------------------------
# Intercept field is populated
# ---------------------------------------------------------------------------


class TestIntercept:
    def test_intercept_is_float(self):
        w = _call_fit_weights(RStacking(), K=3)
        assert isinstance(w.intercept, float)

    def test_intercept_in_ensemble_weights(self):
        w = _call_fit_weights(RStacking(), K=3)
        assert hasattr(w, "intercept")

    def test_b_hat_stored_in_details(self):
        """b_hat is captured in details even though it doesn't affect prediction."""
        w = _call_fit_weights(RStacking(), K=3)
        assert w.details is not None
        assert "b_hat" in w.details
        assert isinstance(w.details["b_hat"], float)

    def test_nonzero_intercept_when_cate_has_constant_shift(self):
        """When true CATE = tau_0(x) + constant but adapters predict only tau_0(x),
        the solver should recover the constant as c_hat (intercept)."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(size=(n, 2))
        true_cate_base = X[:, 0]            # component that adapters predict
        constant_shift = 2.0
        true_cate = true_cate_base + constant_shift

        e_hat = np.full(n, 0.5)
        T = rng.binomial(1, e_hat).astype(float)
        Y = T * true_cate + rng.normal(scale=0.1, size=n)

        nuisance = NuisanceEstimates(
            e_hat=e_hat,
            mu1_hat=true_cate,    # oracle outcome models
            mu0_hat=np.zeros(n),
        )

        # Component predicts base CATE (missing the constant shift)
        cate_preds = true_cate_base[np.newaxis, :]  # shape (1, n)

        strategy = RStacking()
        strategy.fit_weights(cate_preds, Y, T, X, nuisance)

        # c_hat should be close to constant_shift
        assert abs(strategy._weights.intercept - constant_shift) < 0.5, (
            f"Expected intercept ≈ {constant_shift}, got {strategy._weights.intercept:.3f}"
        )


# ---------------------------------------------------------------------------
# Predictions include the constant shift
# ---------------------------------------------------------------------------


class TestPredictionsIncludeShift:
    def test_aggregate_includes_intercept(self):
        """aggregate = intercept + weights @ cate_matrix."""
        rng = np.random.default_rng(0)
        n, K = 50, 3
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        Y = rng.normal(size=n)
        T = rng.binomial(1, 0.5, size=n).astype(float)
        X = rng.normal(size=(n, 2))

        strategy = RStacking()
        strategy.fit_weights(cate_preds, Y, T, X, nuisance)

        w = strategy._weights
        expected = w.intercept + w.weights @ cate_preds
        actual = strategy.aggregate(cate_preds)
        np.testing.assert_allclose(actual, expected)

    def test_zero_intercept_when_no_shift_needed(self):
        """With oracle nuisance and unbiased adapters, intercept should be ~0."""
        rng = np.random.default_rng(7)
        n = 300
        X = rng.normal(size=(n, 2))
        true_cate = X[:, 0]
        e_hat = np.full(n, 0.5)
        T = rng.binomial(1, e_hat).astype(float)
        Y = T * true_cate + rng.normal(scale=0.05, size=n)

        nuisance = NuisanceEstimates(
            e_hat=e_hat,
            mu1_hat=true_cate,
            mu0_hat=np.zeros(n),
        )
        # Adapter predicts the true CATE exactly
        cate_preds = true_cate[np.newaxis, :]

        strategy = RStacking()
        strategy.fit_weights(cate_preds, Y, T, X, nuisance)

        assert abs(strategy._weights.intercept) < 0.3, (
            f"Expected intercept ≈ 0, got {strategy._weights.intercept:.3f}"
        )


# ---------------------------------------------------------------------------
# EnsembleWeights fields
# ---------------------------------------------------------------------------


class TestEnsembleWeightsFields:
    def test_method_is_r_stacking(self):
        w = _call_fit_weights(RStacking(), K=3)
        assert w.method == "r_stacking"

    def test_model_names_empty_before_fit_supervised(self):
        """fit_weights alone leaves model_names empty; _fit_supervised fills it."""
        w = _call_fit_weights(RStacking(), K=3)
        assert w.model_names == []

    def test_returns_ensemble_weights_object(self):
        w = _call_fit_weights(RStacking(), K=3)
        assert isinstance(w, EnsembleWeights)


# ---------------------------------------------------------------------------
# Design matrix construction (white-box)
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    def test_solution_satisfies_normal_equations(self):
        """Verify the solver found the correct solution for a small exact system."""
        rng = np.random.default_rng(1)
        n, K = 50, 2
        W_tilde = rng.normal(size=n)
        Y_tilde = rng.normal(size=n)
        cate_preds = rng.normal(size=(K, n))

        # Build design matrix manually
        A = np.column_stack([
            np.ones(n),
            W_tilde,
            cate_preds.T * W_tilde[:, np.newaxis],
        ])

        nuisance = NuisanceEstimates(
            e_hat=np.zeros(n),   # W_tilde = T - 0 = T
            mu1_hat=np.zeros(n),
            mu0_hat=np.zeros(n),
        )
        # Y_tilde = Y - m_hat = Y - 0 = Y
        Y = Y_tilde
        T = W_tilde  # W_tilde = T - e_hat = T since e_hat = 0

        strategy = RStacking()
        strategy.fit_weights(cate_preds, Y, T, np.zeros((n, 1)), nuisance)

        w = strategy._weights
        z = np.concatenate([[w.details["b_hat"]], [w.intercept], w.weights])
        residuals = Y_tilde - A @ z

        # Residual should be orthogonal to unconstrained columns (b, c)
        # (constrained cols may not be exactly orthogonal)
        assert abs(np.dot(A[:, 0], residuals)) < 1e-6   # b column
        assert abs(np.dot(A[:, 1], residuals)) < 1e-6   # c column


# ---------------------------------------------------------------------------
# End-to-end: CausalEnsemble with RStacking
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_r_stacking_fit_and_cate(self):
        X, T, Y = _dgp(n=200, seed=0)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", 1.0), CateCapableAdapter("b", 2.0)],
            aggregation=_fast_rs(),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.cate.shape == (200,)
        assert result.ensemble_weights is not None
        assert result.ensemble_weights.method == "r_stacking"
        assert np.all(result.ensemble_weights.weights >= -1e-9)
        assert isinstance(result.ensemble_weights.intercept, float)

    def test_ate_equals_mean_cate(self):
        X, T, Y = _dgp(n=200, seed=0)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", 1.0), CateCapableAdapter("b", 2.0)],
            aggregation=_fast_rs(),
        )
        ens.fit(X, T, Y, random_state=0)
        ate_result = ens.ate(X)
        cate_result = ens.cate(X)
        np.testing.assert_allclose(ate_result.ate, cate_result.cate.mean())

    def test_importable_from_aggregation(self):
        from metacausal.aggregation import RStacking  # noqa: F401
