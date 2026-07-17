"""Tests for B2.3 (injectable nuisance) and B2.4 (injectable pseudo-outcomes)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal.aggregation import CausalStacking, CrossFitSplit, QAggregation
from metacausal.aggregation.base import SupervisedStrategy
from metacausal.aggregation.nuisance import NuisanceEstimates, fit_nuisance

from tests.test_supervised_fit import CateCapableAdapter, _dgp, UniformSupervisedStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_strategy_cls(cls, **kwargs):
    return cls(
        split=CrossFitSplit(n_folds=3, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
        **kwargs,
    )


def _call_fit_weights(strategy, n=100, K=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = rng.normal(size=n)
    cate_preds = rng.normal(size=(K, n))
    nuisance = NuisanceEstimates(
        e_hat=np.full(n, 0.5),
        mu1_hat=np.zeros(n),
        mu0_hat=np.zeros(n),
    )
    strategy.fit_weights(cate_preds, Y, T, X, nuisance)
    return strategy._weights


# ---------------------------------------------------------------------------
# B2.3: fit_nuisance_fn on SupervisedStrategy
# ---------------------------------------------------------------------------


class TestInjectableNuisance:
    def test_default_none_uses_standard_fit_nuisance(self):
        """Default fit_nuisance_fn=None: behaviour unchanged."""
        strategy = _fast_strategy_cls(CausalStacking)
        assert strategy.fit_nuisance_fn is None

    def test_fit_nuisance_fn_field_on_base_class(self):
        """All supervised strategies inherit fit_nuisance_fn."""
        for cls in (CausalStacking, QAggregation):
            s = cls()
            assert hasattr(s, "fit_nuisance_fn")
            assert s.fit_nuisance_fn is None

    def test_custom_fit_nuisance_fn_is_called(self):
        """Custom fit_nuisance_fn is called instead of the default."""
        from metacausal import CausalEnsemble

        call_log = []

        def tracking_nuisance_fn(X, T, Y, fold_spec,
                                  propensity_model, outcome_model,
                                  propensity_trim, **kwargs):
            call_log.append({"n": len(T)})
            # Delegate to real function so the pipeline completes
            return fit_nuisance(X, T, Y, fold_spec,
                                propensity_model=propensity_model,
                                outcome_model=outcome_model,
                                propensity_trim=propensity_trim,
                                **kwargs)

        X, T, Y = _dgp(n=200, seed=0)
        strategy = CausalStacking(
            split=CrossFitSplit(n_folds=3, stratify=False),
            propensity_model=LogisticRegression(max_iter=200),
            outcome_model=LinearRegression(),
            fit_nuisance_fn=tracking_nuisance_fn,
        )
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=strategy,
        )
        ens.fit(X, T, Y, random_state=0)

        # tracking_nuisance_fn should have been called once per pipeline
        assert len(call_log) >= 1, "Custom fit_nuisance_fn was never called"

    def test_custom_nuisance_fn_can_return_constant_estimates(self):
        """Custom fn can return any valid NuisanceEstimates."""
        from metacausal import CausalEnsemble

        def constant_nuisance(X, T, Y, fold_spec,
                              propensity_model, outcome_model, propensity_trim, **kwargs):
            n = len(T)
            return NuisanceEstimates(
                e_hat=np.full(n, 0.5),
                mu1_hat=np.zeros(n),
                mu0_hat=np.zeros(n),
            )

        X, T, Y = _dgp(n=200, seed=0)
        strategy = CausalStacking(
            split=CrossFitSplit(n_folds=3, stratify=False),
            propensity_model=None,  # irrelevant — custom fn ignores it
            outcome_model=None,
            fit_nuisance_fn=constant_nuisance,
        )
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=strategy,
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.cate.shape == (200,)

    def test_custom_nuisance_fn_affects_weights(self):
        """Different nuisance → different weights."""
        from metacausal import CausalEnsemble

        def nuisance_A(X, T, Y, fold_spec, propensity_model, outcome_model, propensity_trim, **kwargs):
            n = len(T)
            return NuisanceEstimates(e_hat=np.full(n, 0.3), mu1_hat=np.ones(n), mu0_hat=np.zeros(n))

        def nuisance_B(X, T, Y, fold_spec, propensity_model, outcome_model, propensity_trim, **kwargs):
            n = len(T)
            return NuisanceEstimates(e_hat=np.full(n, 0.7), mu1_hat=np.zeros(n), mu0_hat=np.ones(n))

        X, T, Y = _dgp(n=200, seed=0)
        methods = [CateCapableAdapter("a", scale=1.0), CateCapableAdapter("b", scale=2.0)]

        ens_a = CausalEnsemble(
            methods=methods,
            aggregation=CausalStacking(
                split=CrossFitSplit(n_folds=3, stratify=False),
                fit_nuisance_fn=nuisance_A,
            ),
        )
        ens_b = CausalEnsemble(
            methods=methods,
            aggregation=CausalStacking(
                split=CrossFitSplit(n_folds=3, stratify=False),
                fit_nuisance_fn=nuisance_B,
            ),
        )
        ens_a.fit(X, T, Y, random_state=0)
        ens_b.fit(X, T, Y, random_state=0)

        w_a = ens_a.aggregation.ensemble_weights.weights
        w_b = ens_b.aggregation.ensemble_weights.weights
        # Different nuisance → different pseudo-outcomes → different weights
        assert not np.allclose(w_a, w_b, atol=1e-6), \
            "Different nuisance functions should produce different weights"


# ---------------------------------------------------------------------------
# B2.4: pseudo_outcome_fn on CausalStacking and QAggregation
# ---------------------------------------------------------------------------


class TestInjectablePseudoOutcome:
    def test_default_none_uses_dr_pseudo_outcome(self):
        for cls in (CausalStacking, QAggregation):
            s = cls()
            assert hasattr(s, "pseudo_outcome_fn")
            assert s.pseudo_outcome_fn is None

    def test_r_stacking_has_no_pseudo_outcome_fn(self):
        """RStacking uses Robinson residuals, not DR pseudo-outcomes."""
        from metacausal.aggregation import RStacking
        s = RStacking()
        assert not hasattr(s, "pseudo_outcome_fn")

    def test_custom_pseudo_outcome_fn_is_called_causal_stacking(self):
        call_log = []

        def tracking_po(Y, T, nuisance):
            call_log.append(len(Y))
            from metacausal.aggregation.nuisance import dr_pseudo_outcome
            return dr_pseudo_outcome(Y, T, nuisance)

        w = _call_fit_weights(CausalStacking(pseudo_outcome_fn=tracking_po))
        assert len(call_log) >= 1, "Custom pseudo_outcome_fn was never called"

    def test_custom_pseudo_outcome_fn_is_called_q_agg(self):
        call_log = []

        def tracking_po(Y, T, nuisance):
            call_log.append(len(Y))
            from metacausal.aggregation.nuisance import dr_pseudo_outcome
            return dr_pseudo_outcome(Y, T, nuisance)

        w = _call_fit_weights(QAggregation(pseudo_outcome_fn=tracking_po))
        assert len(call_log) >= 1, "Custom pseudo_outcome_fn was never called"

    def test_custom_pseudo_outcome_affects_weights(self):
        """Replacing DR pseudo-outcome with a constant changes the weights."""
        rng = np.random.default_rng(5)
        n, K = 150, 3
        X = rng.normal(size=(n, 2))
        T = rng.binomial(1, 0.5, size=n).astype(float)
        Y = rng.normal(size=n)
        cate_preds = rng.normal(size=(K, n))
        nuisance = NuisanceEstimates(
            e_hat=np.full(n, 0.5),
            mu1_hat=np.zeros(n),
            mu0_hat=np.zeros(n),
        )

        # Default DR pseudo-outcome
        s_default = CausalStacking()
        s_default.fit_weights(cate_preds, Y, T, X, nuisance)
        w_default = s_default._weights.weights.copy()

        # Constant pseudo-outcome (zeros) → different loss landscape
        s_custom = CausalStacking(pseudo_outcome_fn=lambda Y, T, nu: np.zeros(len(Y)))
        s_custom.fit_weights(cate_preds, Y, T, X, nuisance)
        w_custom = s_custom._weights.weights.copy()

        assert not np.allclose(w_default, w_custom, atol=1e-4), \
            "Custom pseudo-outcome function should produce different weights"

    def test_custom_pseudo_outcome_works_for_q_agg(self):
        """QAggregation also respects pseudo_outcome_fn."""
        rng = np.random.default_rng(9)
        n, K = 100, 3
        X = rng.normal(size=(n, 2))
        T = rng.binomial(1, 0.5, size=n).astype(float)
        Y = rng.normal(size=n)
        cate_preds = rng.normal(size=(K, n))
        nuisance = NuisanceEstimates(
            e_hat=np.full(n, 0.5), mu1_hat=np.zeros(n), mu0_hat=np.zeros(n)
        )

        custom_po = lambda Y, T, nu: np.ones(len(Y))  # all-ones pseudo-outcome
        s = QAggregation(nu=0.5, pseudo_outcome_fn=custom_po)
        s.fit_weights(cate_preds, Y, T, X, nuisance)
        # Weights are on simplex and non-negative — custom fn didn't break anything
        assert abs(s._weights.weights.sum() - 1.0) < 1e-6
        assert np.all(s._weights.weights >= -1e-9)

    def test_pseudo_outcome_fn_not_repr(self):
        """Callable fields don't clutter repr."""
        s = CausalStacking(pseudo_outcome_fn=lambda Y, T, nu: np.zeros(len(Y)))
        rep = repr(s)
        assert "pseudo_outcome_fn" not in rep

    def test_fit_nuisance_fn_not_repr(self):
        s = CausalStacking(fit_nuisance_fn=lambda *a, **kw: None)
        rep = repr(s)
        assert "fit_nuisance_fn" not in rep
