"""Tests for Select."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal import CausalEnsemble
from metacausal.aggregation import CrossFitSplit, QAggregation, Select
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


def _fast_select(**kwargs):
    return Select(
        split=CrossFitSplit(n_folds=3, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
        **kwargs,
    )


def _dominant_model_data(n: int = 300, K: int = 3, seed: int = 7):
    """DR-pseudo-outcome-aligned data where model 0 is the clear DR-risk winner."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    e_hat = np.full(n, 0.5)
    T = rng.binomial(1, e_hat).astype(float)
    true_cate = rng.normal(size=n)
    Y = T * true_cate + rng.normal(scale=0.01, size=n)
    nuisance = NuisanceEstimates(e_hat=e_hat, mu1_hat=np.zeros(n), mu0_hat=np.zeros(n))
    cate_predictions = np.stack([
        true_cate,                                    # model 0: perfect
        rng.normal(size=n),                            # model 1: noise
        rng.normal(size=n),                            # model 2: noise
    ], axis=0)
    return X, T, Y, nuisance, cate_predictions


# ---------------------------------------------------------------------------
# fit_weights: interface and EnsembleWeights
# ---------------------------------------------------------------------------


class TestFitWeightsInterface:
    def test_returns_ensemble_weights(self):
        n, K = 80, 3
        rng = np.random.default_rng(0)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        strategy = Select()
        w = strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, rng.normal(size=(n, 2)), nuisance)
        assert isinstance(w, EnsembleWeights)
        assert w.method == "select_dr"

    def test_weights_are_one_hot(self):
        n, K = 80, 4
        rng = np.random.default_rng(0)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        strategy = Select()
        w = strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, rng.normal(size=(n, 2)), nuisance)
        assert w.weights.shape == (K,)
        assert np.count_nonzero(w.weights) == 1
        assert w.weights[np.argmax(w.weights)] == 1.0

    def test_risk_stored_in_details(self):
        n, K = 80, 3
        rng = np.random.default_rng(0)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        strategy = Select()
        strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, rng.normal(size=(n, 2)), nuisance)
        assert strategy._weights.details is not None
        assert strategy._weights.details["risk"].shape == (K,)

    def test_k1_edge_case(self):
        n = 50
        nuisance = _oracle_nuisance(n)
        cate_preds = np.ones((1, n))
        strategy = Select()
        strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, np.zeros((n, 2)), nuisance)
        np.testing.assert_allclose(strategy._weights.weights, [1.0])

    def test_invalid_loss_raises(self):
        n, K = 50, 3
        rng = np.random.default_rng(0)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        strategy = Select(loss="bogus")
        with pytest.raises(ValueError, match="loss must be 'dr' or 'r'"):
            strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, np.zeros((n, 2)), nuisance)


# ---------------------------------------------------------------------------
# loss="dr": picks the lowest-DR-risk model
# ---------------------------------------------------------------------------


class TestDrLoss:
    def test_picks_dominant_model(self):
        X, T, Y, nuisance, cate_preds = _dominant_model_data()
        strategy = Select(loss="dr")
        w = strategy.fit_weights(cate_preds, Y, T, X, nuisance)
        assert np.argmax(w.weights) == 0

    def test_matches_q_aggregation_nu1_argmax(self):
        """Select(loss='dr') picks the same model as QAggregation(nu=1)'s vertex."""
        X, T, Y, nuisance, cate_preds = _dominant_model_data(seed=3)
        w_select = Select(loss="dr").fit_weights(cate_preds, Y, T, X, nuisance)
        w_qagg = QAggregation(nu=1.0, beta=0.0).fit_weights(cate_preds, Y, T, X, nuisance)
        assert np.argmax(w_select.weights) == np.argmax(w_qagg.weights)


# ---------------------------------------------------------------------------
# loss="r": picks the lowest-R-risk model
# ---------------------------------------------------------------------------


class TestRLoss:
    def test_picks_dominant_model(self):
        rng = np.random.default_rng(11)
        n = 300
        X = rng.normal(size=(n, 2))
        e_hat = np.full(n, 0.5)
        T = rng.binomial(1, e_hat).astype(float)
        m_hat = rng.normal(size=n)
        true_cate = rng.normal(size=n)
        Y = m_hat + (T - e_hat) * true_cate + rng.normal(scale=0.01, size=n)
        nuisance = NuisanceEstimates(e_hat=e_hat, mu1_hat=m_hat + 0.5 * true_cate, mu0_hat=m_hat - 0.5 * true_cate)

        cate_preds = np.stack([
            true_cate,                 # model 0: perfect
            rng.normal(size=n),        # model 1: noise
            rng.normal(size=n),        # model 2: noise
        ], axis=0)

        strategy = Select(loss="r")
        w = strategy.fit_weights(cate_preds, Y, T, X, nuisance)
        assert np.argmax(w.weights) == 0

    def test_risk_matches_manual_robinson_computation(self):
        n, K = 100, 3
        rng = np.random.default_rng(5)
        X = rng.normal(size=(n, 2))
        T = rng.binomial(1, 0.5, size=n).astype(float)
        Y = rng.normal(size=n)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)

        strategy = Select(loss="r")
        w = strategy.fit_weights(cate_preds, Y, T, X, nuisance)

        Y_tilde, W_tilde = robinson_residuals(Y, T, nuisance)
        expected_risk = np.mean((Y_tilde[np.newaxis, :] - cate_preds * W_tilde[np.newaxis, :]) ** 2, axis=1)
        np.testing.assert_allclose(w.details["risk"], expected_risk)
        assert np.argmax(w.weights) == int(np.argmin(expected_risk))


# ---------------------------------------------------------------------------
# End-to-end: CausalEnsemble with real supervised fit path
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_fit_and_cate(self):
        X, T, Y = _dgp(n=200, seed=0)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", 1.0), CateCapableAdapter("b", 2.0)],
            aggregation=_fast_select(),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.cate.shape == (200,)
        assert result.ensemble_weights is not None
        assert result.ensemble_weights.method == "select_dr"
        assert np.count_nonzero(result.ensemble_weights.weights) == 1

    def test_fit_and_cate_r_loss(self):
        X, T, Y = _dgp(n=200, seed=0)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", 1.0), CateCapableAdapter("b", 2.0)],
            aggregation=_fast_select(loss="r"),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.ensemble_weights.method == "select_r"

    def test_importable_from_aggregation(self):
        from metacausal.aggregation import Select  # noqa: F401
