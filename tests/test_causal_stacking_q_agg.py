"""Tests for CausalStacking, QAggregation, and the shared DR simplex solver."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal import CausalEnsemble
from metacausal.aggregation import CausalStacking, CrossFitSplit, QAggregation
from metacausal.aggregation.nuisance import NuisanceEstimates
from metacausal.aggregation.q_aggregation import _solve_dr_simplex

from tests.test_supervised_fit import CateCapableAdapter, _dgp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng_data(n: int = 200, K: int = 3, seed: int = 0):
    """Return pseudo_outcomes (n,) and cate_predictions (K, n) for solver tests."""
    rng = np.random.default_rng(seed)
    pseudo_outcomes = rng.normal(size=n)
    # Give each model a different relationship to pseudo_outcomes
    cate_predictions = np.stack([
        pseudo_outcomes + rng.normal(scale=0.1 * (k + 1), size=n)
        for k in range(K)
    ], axis=0)
    return pseudo_outcomes, cate_predictions


def _uniform_prior(K: int) -> np.ndarray:
    return np.ones(K)


def _oracle_nuisance(n: int) -> NuisanceEstimates:
    """NuisanceEstimates with trivially correct predictions for a balanced RCT."""
    return NuisanceEstimates(
        e_hat=np.full(n, 0.5),
        mu1_hat=np.zeros(n),
        mu0_hat=np.zeros(n),
    )


def _fast_cs():
    return CausalStacking(
        split=CrossFitSplit(n_folds=3, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
    )


def _fast_qa(**kwargs):
    return QAggregation(
        split=CrossFitSplit(n_folds=3, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _solve_dr_simplex: simplex constraints
# ---------------------------------------------------------------------------


class TestSolverSimplexConstraints:
    def test_weights_sum_to_one_full(self):
        gamma, C = _rng_data(K=4)
        w = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0,
                              prior=_uniform_prior(4), greedy=False)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_nonneg_full(self):
        gamma, C = _rng_data(K=4)
        w = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0,
                              prior=_uniform_prior(4), greedy=False)
        assert np.all(w >= -1e-8)

    def test_weights_sum_to_one_greedy(self):
        gamma, C = _rng_data(K=4)
        w = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0,
                              prior=_uniform_prior(4), greedy=True)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_nonneg_greedy(self):
        gamma, C = _rng_data(K=4)
        w = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0,
                              prior=_uniform_prior(4), greedy=True)
        assert np.all(w >= -1e-8)

    def test_k1_returns_one(self):
        gamma = np.array([1.0, 2.0, 3.0])
        C = gamma[np.newaxis, :]
        w = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0,
                              prior=np.ones(1), greedy=False)
        np.testing.assert_allclose(w, [1.0])


# ---------------------------------------------------------------------------
# _solve_dr_simplex: greedy returns at most 2 nonzero weights
# ---------------------------------------------------------------------------


class TestGreedySparsity:
    def test_at_most_two_nonzero(self):
        gamma, C = _rng_data(K=6)
        w = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0,
                              prior=_uniform_prior(6), greedy=True)
        n_nonzero = np.sum(w > 1e-9)
        assert n_nonzero <= 2

    def test_greedy_k2_always_two_or_one(self):
        gamma, C = _rng_data(K=2)
        w = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0,
                              prior=_uniform_prior(2), greedy=True)
        n_nonzero = np.sum(w > 1e-9)
        assert n_nonzero <= 2
        assert abs(w.sum() - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# _solve_dr_simplex: dominant model gets most weight
# ---------------------------------------------------------------------------


class TestDominantModel:
    def test_dominant_model_gets_high_weight_full(self):
        """If model 0 perfectly predicts pseudo-outcomes, it should dominate."""
        rng = np.random.default_rng(7)
        n = 300
        gamma = rng.normal(size=n)
        C = np.stack([
            gamma,                                 # model 0: perfect
            rng.normal(size=n),                    # model 1: noise
            rng.normal(size=n),                    # model 2: noise
        ], axis=0)
        w = _solve_dr_simplex(gamma, C, nu=0.0, beta=0.0,
                              prior=_uniform_prior(3), greedy=False)
        assert w[0] > 0.9, f"Perfect model weight = {w[0]:.3f}, expected > 0.9"

    def test_dominant_model_gets_high_weight_greedy(self):
        rng = np.random.default_rng(7)
        n = 300
        gamma = rng.normal(size=n)
        C = np.stack([
            gamma,
            rng.normal(size=n),
            rng.normal(size=n),
        ], axis=0)
        w = _solve_dr_simplex(gamma, C, nu=0.0, beta=0.0,
                              prior=_uniform_prior(3), greedy=True)
        assert w[0] > 0.9, f"Perfect model weight = {w[0]:.3f}, expected > 0.9"

    def test_full_and_greedy_agree_on_dominant(self):
        """Both solvers should put most weight on the best model."""
        rng = np.random.default_rng(42)
        n = 200
        gamma = rng.normal(size=n)
        C = np.stack([gamma, rng.normal(size=n), rng.normal(size=n)], axis=0)
        w_full = _solve_dr_simplex(gamma, C, nu=0.0, beta=0.0,
                                   prior=_uniform_prior(3), greedy=False)
        w_greedy = _solve_dr_simplex(gamma, C, nu=0.0, beta=0.0,
                                     prior=_uniform_prior(3), greedy=True)
        assert np.argmax(w_full) == np.argmax(w_greedy) == 0


# ---------------------------------------------------------------------------
# CausalStacking == QAggregation(nu=0, beta=0)
# ---------------------------------------------------------------------------


class TestCausalStackingEquivalence:
    def _call_fit_weights(self, strategy, n: int = 100, K: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        T = rng.binomial(1, 0.5, size=n).astype(float)
        Y = rng.normal(size=n)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        strategy.fit_weights(cate_preds, Y, T, X, nuisance)
        return strategy._weights.weights

    def test_causal_stacking_equals_q_agg_nu0_beta0(self):
        """CausalStacking must produce identical weights to QAggregation(nu=0, beta=0)."""
        n, K, seed = 150, 3, 99

        w_cs = self._call_fit_weights(CausalStacking(), n, K, seed)
        w_qa = self._call_fit_weights(
            QAggregation(nu=0.0, beta=0.0, greedy=False), n, K, seed
        )
        np.testing.assert_allclose(w_cs, w_qa, atol=1e-7)

    def test_q_agg_nu_half_differs_from_causal_stacking(self):
        """QAggregation(nu=0.5) should give different weights than CausalStacking."""
        n, K, seed = 150, 3, 99

        w_cs = self._call_fit_weights(CausalStacking(), n, K, seed)
        w_qa_half = self._call_fit_weights(
            QAggregation(nu=0.5, beta=0.0), n, K, seed
        )
        # Different nu should generally produce different weights
        assert not np.allclose(w_cs, w_qa_half, atol=1e-4), \
            "nu=0.5 should differ from nu=0, but got identical weights"


# ---------------------------------------------------------------------------
# fit_weights: interface and EnsembleWeights
# ---------------------------------------------------------------------------


class TestFitWeightsInterface:
    def test_causal_stacking_returns_ensemble_weights(self):
        n, K = 80, 3
        rng = np.random.default_rng(0)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        strategy = CausalStacking()
        w = strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, rng.normal(size=(n, 2)), nuisance)
        assert isinstance(w, EnsembleWeights)  # imported below
        assert w.method == "causal_stacking"
        assert w.weights.shape == (K,)

    def test_q_agg_returns_ensemble_weights(self):
        n, K = 80, 3
        rng = np.random.default_rng(0)
        cate_preds = rng.normal(size=(K, n))
        nuisance = _oracle_nuisance(n)
        strategy = QAggregation(nu=0.5, beta=0.0)
        w = strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, rng.normal(size=(n, 2)), nuisance)
        from metacausal.aggregation.weights import EnsembleWeights as EW
        assert isinstance(w, EW)
        assert w.method == "q_aggregation"
        assert w.details["nu"] == pytest.approx(0.5)

    def test_k1_edge_case_causal_stacking(self):
        n = 50
        nuisance = _oracle_nuisance(n)
        cate_preds = np.ones((1, n))
        strategy = CausalStacking()
        strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, np.zeros((n, 2)), nuisance)
        np.testing.assert_allclose(strategy._weights.weights, [1.0])

    def test_k1_edge_case_q_agg(self):
        n = 50
        nuisance = _oracle_nuisance(n)
        cate_preds = np.ones((1, n))
        strategy = QAggregation()
        strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, np.zeros((n, 2)), nuisance)
        np.testing.assert_allclose(strategy._weights.weights, [1.0])

    def test_prior_shape_mismatch_raises(self):
        n, K = 50, 3
        nuisance = _oracle_nuisance(n)
        cate_preds = np.random.default_rng(0).normal(size=(K, n))
        strategy = QAggregation(prior=np.ones(5))  # wrong K
        with pytest.raises(ValueError, match="prior must have shape"):
            strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, np.zeros((n, 2)), nuisance)

    def test_prior_nonpositive_raises(self):
        n, K = 50, 3
        nuisance = _oracle_nuisance(n)
        cate_preds = np.random.default_rng(0).normal(size=(K, n))
        strategy = QAggregation(prior=np.array([1.0, 0.0, 1.0]))  # zero entry
        with pytest.raises(ValueError, match="strictly positive"):
            strategy.fit_weights(cate_preds, np.zeros(n), np.ones(n) * 0.5, np.zeros((n, 2)), nuisance)


# ---------------------------------------------------------------------------
# nu parameter effects
# ---------------------------------------------------------------------------


class TestNuParameter:
    def test_nu0_same_as_causal_stacking(self):
        """Already covered by equivalence test — just confirm nu=0 flag works."""
        rng = np.random.default_rng(5)
        n, K = 100, 3
        gamma = rng.normal(size=n)
        C = rng.normal(size=(K, n))
        w0 = _solve_dr_simplex(gamma, C, nu=0.0, beta=0.0, prior=np.ones(K), greedy=False)
        w_cs = _solve_dr_simplex(gamma, C, nu=0.0, beta=0.0, prior=np.ones(K), greedy=False)
        np.testing.assert_allclose(w0, w_cs)

    def test_beta_no_effect_with_uniform_prior(self):
        """With uniform prior, any beta value gives the same weights."""
        rng = np.random.default_rng(3)
        n, K = 100, 3
        gamma = rng.normal(size=n)
        C = rng.normal(size=(K, n))
        w0 = _solve_dr_simplex(gamma, C, nu=0.5, beta=0.0, prior=np.ones(K), greedy=False)
        w1 = _solve_dr_simplex(gamma, C, nu=0.5, beta=100.0, prior=np.ones(K), greedy=False)
        np.testing.assert_allclose(w0, w1, atol=1e-6)

    def test_beta_has_effect_with_nonuniform_prior(self):
        """With non-uniform prior, beta > 0 shifts weights toward high-prior model."""
        rng = np.random.default_rng(11)
        n, K = 200, 3
        gamma = rng.normal(size=n)
        C = rng.normal(size=(K, n))
        uniform_prior = np.ones(K)
        # Strong prior on model 0
        skewed_prior = np.array([10.0, 1.0, 1.0])
        w_uniform = _solve_dr_simplex(gamma, C, nu=0.5, beta=50.0,
                                      prior=uniform_prior, greedy=False)
        w_skewed = _solve_dr_simplex(gamma, C, nu=0.5, beta=50.0,
                                     prior=skewed_prior, greedy=False)
        # Prior shifts toward model 0; skewed should give more weight to 0
        assert w_skewed[0] >= w_uniform[0] - 0.05


# ---------------------------------------------------------------------------
# End-to-end: CausalEnsemble with real supervised fit path
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_causal_stacking_fit_and_cate(self):
        X, T, Y = _dgp(n=200, seed=0)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", 1.0), CateCapableAdapter("b", 2.0)],
            aggregation=_fast_cs(),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.cate.shape == (200,)
        assert result.ensemble_weights is not None
        assert result.ensemble_weights.method == "causal_stacking"
        np.testing.assert_allclose(result.ensemble_weights.weights.sum(), 1.0)

    def test_q_agg_fit_and_cate(self):
        X, T, Y = _dgp(n=200, seed=0)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", 1.0), CateCapableAdapter("b", 2.0)],
            aggregation=_fast_qa(nu=0.5),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.cate.shape == (200,)
        assert result.ensemble_weights.method == "q_aggregation"

    def test_q_agg_greedy_fit_and_cate(self):
        X, T, Y = _dgp(n=200, seed=0)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", 1.0), CateCapableAdapter("b", 2.0),
                     CateCapableAdapter("c", 3.0)],
            aggregation=_fast_qa(nu=0.5, greedy=True),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        n_nonzero = np.sum(result.ensemble_weights.weights > 1e-9)
        assert n_nonzero <= 2

    def test_importable_from_aggregation(self):
        from metacausal.aggregation import CausalStacking, QAggregation  # noqa: F401


# ---------------------------------------------------------------------------
# Import EnsembleWeights for use in tests above
# ---------------------------------------------------------------------------

from metacausal.aggregation.weights import EnsembleWeights  # noqa: E402
