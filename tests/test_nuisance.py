"""Tests for nuisance model fitting and pseudo-outcome utilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal.aggregation.nuisance import (
    NuisanceEstimates,
    dr_pseudo_outcome,
    fit_nuisance,
    robinson_residuals,
)
from metacausal.aggregation.splitting import CrossFitSplit, TrainAvgSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dgp(n: int = 300, seed: int = 0):
    """Simple DGP with known propensity and outcome functions.

    Propensity: e(x) = sigmoid(x[:, 0])
    mu1(x) = 2 * x[:, 0] + x[:, 1]
    mu0(x) = x[:, 0]
    tau(x) = mu1 - mu0 = x[:, 0] + x[:, 1]    (CATE)
    Y = mu0 + T * tau + noise
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logit = X[:, 0]
    e_true = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, e_true).astype(float)
    mu1_true = 2 * X[:, 0] + X[:, 1]
    mu0_true = X[:, 0]
    Y = np.where(T == 1, mu1_true, mu0_true) + rng.normal(scale=0.1, size=n)
    return X, T, Y, e_true, mu1_true, mu0_true


def _fast_models():
    """Lightweight sklearn models for use in tests (no LightGBM)."""
    return (
        LogisticRegression(max_iter=500, random_state=0),
        LinearRegression(),
    )


def _oracle_nuisance(X, T, Y, e_true, mu1_true, mu0_true, trim=0.01):
    """NuisanceEstimates populated with the true (oracle) nuisance functions."""
    e = np.clip(e_true, trim, 1 - trim)
    return NuisanceEstimates(
        e_hat=e,
        mu1_hat=mu1_true,
        mu0_hat=mu0_true,
    )


# ---------------------------------------------------------------------------
# NuisanceEstimates
# ---------------------------------------------------------------------------


class TestNuisanceEstimates:
    def test_m_hat_formula(self):
        e = np.array([0.3, 0.6, 0.8])
        mu1 = np.array([1.0, 2.0, 3.0])
        mu0 = np.array([0.0, 1.0, 2.0])
        nu = NuisanceEstimates(e_hat=e, mu1_hat=mu1, mu0_hat=mu0)
        expected = e * mu1 + (1 - e) * mu0
        np.testing.assert_allclose(nu.m_hat, expected)


# ---------------------------------------------------------------------------
# fit_nuisance: CrossFitSplit
# ---------------------------------------------------------------------------


class TestFitNuisanceCrossFit:
    def setup_method(self):
        self.X, self.T, self.Y, self.e_true, self.mu1_true, self.mu0_true = _make_dgp(n=200)
        self.prop_model, self.out_model = _fast_models()
        self.fold_spec = CrossFitSplit(n_folds=5, stratify=True).split(self.T, random_state=0)

    def test_returns_nuisance_estimates(self):
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
        )
        assert isinstance(nu, NuisanceEstimates)

    def test_full_coverage_no_nan(self):
        """CrossFitSplit: all n positions populated, no NaN."""
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
        )
        assert not np.any(np.isnan(nu.e_hat))
        assert not np.any(np.isnan(nu.mu1_hat))
        assert not np.any(np.isnan(nu.mu0_hat))
        assert len(nu.e_hat) == len(self.T)

    def test_propensity_clipped(self):
        """e_hat values lie within [trim, 1-trim]."""
        trim = 0.01
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
            propensity_trim=trim,
        )
        assert np.all(nu.e_hat >= trim)
        assert np.all(nu.e_hat <= 1 - trim)

    def test_custom_trim(self):
        """Custom trim value is respected."""
        trim = 0.05
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
            propensity_trim=trim,
        )
        assert np.all(nu.e_hat >= trim)
        assert np.all(nu.e_hat <= 1 - trim)

    def test_oof_property(self):
        """Each observation's prediction is not made by a model trained on it.

        Proxy test: propensity predictions on held-out folds should have higher
        variance than in-sample predictions (if we refitted on the full data).
        More directly: predictions differ from a single full-data fit.
        """
        nu_oof = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
        )
        # Fit once on full data
        full_prop = LogisticRegression(max_iter=500, random_state=0)
        full_prop.fit(self.X, self.T)
        full_e = full_prop.predict_proba(self.X)[:, 1]
        # OOF predictions should differ from in-sample full-data predictions
        assert not np.allclose(nu_oof.e_hat, np.clip(full_e, 0.01, 0.99))

    def test_m_hat_derived_correctly(self):
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
        )
        expected_m_hat = nu.e_hat * nu.mu1_hat + (1 - nu.e_hat) * nu.mu0_hat
        np.testing.assert_allclose(nu.m_hat, expected_m_hat)

    def test_default_models_used_when_none(self):
        """fit_nuisance runs without error when models are None (uses defaults)."""
        # Use a minimal fold_spec with LinearRegression/LogisticRegression internally
        # to avoid needing LightGBM — we monkeypatch defaults for the test.
        from unittest.mock import patch
        from metacausal import defaults

        with (
            patch.object(defaults, "default_propensity_model", return_value=LogisticRegression(max_iter=200)),
            patch.object(defaults, "default_outcome_model", return_value=LinearRegression()),
        ):
            nu = fit_nuisance(self.X, self.T, self.Y, self.fold_spec)
        assert not np.any(np.isnan(nu.e_hat))


# ---------------------------------------------------------------------------
# fit_nuisance: TrainAvgSplit
# ---------------------------------------------------------------------------


class TestFitNuisanceTrainAvg:
    def setup_method(self):
        self.X, self.T, self.Y, self.e_true, self.mu1_true, self.mu0_true = _make_dgp(n=200)
        self.prop_model, self.out_model = _fast_models()
        self.fold_spec = TrainAvgSplit(avg_frac=0.25, stratify=True).split(self.T, random_state=0)
        self.avg_idx = self.fold_spec.test_indices[0]
        self.train_idx = self.fold_spec.train_indices[0]

    def test_averaging_set_populated(self):
        """Averaging-set positions are filled (not NaN)."""
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
        )
        assert not np.any(np.isnan(nu.e_hat[self.avg_idx]))
        assert not np.any(np.isnan(nu.mu1_hat[self.avg_idx]))
        assert not np.any(np.isnan(nu.mu0_hat[self.avg_idx]))

    def test_training_set_is_nan(self):
        """Training-set positions are left as NaN (not used for weight optimization)."""
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
        )
        assert np.all(np.isnan(nu.e_hat[self.train_idx]))
        assert np.all(np.isnan(nu.mu1_hat[self.train_idx]))
        assert np.all(np.isnan(nu.mu0_hat[self.train_idx]))

    def test_output_length_is_n(self):
        """NuisanceEstimates arrays have length n, not avg_n."""
        n = len(self.T)
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
        )
        assert len(nu.e_hat) == n
        assert len(nu.mu1_hat) == n
        assert len(nu.mu0_hat) == n

    def test_propensity_clipped_in_avg_set(self):
        trim = 0.01
        nu = fit_nuisance(
            self.X, self.T, self.Y, self.fold_spec,
            propensity_model=self.prop_model,
            outcome_model=self.out_model,
            propensity_trim=trim,
        )
        e_avg = nu.e_hat[self.avg_idx]
        assert np.all(e_avg >= trim)
        assert np.all(e_avg <= 1 - trim)


# ---------------------------------------------------------------------------
# fit_nuisance: error conditions
# ---------------------------------------------------------------------------


class TestFitNuisanceErrors:
    def test_no_treated_units_in_fold_raises(self):
        """Empty treated strata in a training fold raises ValueError."""
        # All control: no treated units anywhere
        n = 40
        X = np.random.default_rng(0).normal(size=(n, 2))
        T = np.zeros(n)
        Y = np.random.default_rng(0).normal(size=n)
        fold_spec = CrossFitSplit(n_folds=2, stratify=False).split(T, random_state=0)
        prop_model, out_model = _fast_models()
        with pytest.raises(ValueError, match="no treated units"):
            fit_nuisance(X, T, Y, fold_spec, propensity_model=prop_model, outcome_model=out_model)

    def test_no_control_units_in_fold_raises(self):
        """Empty control strata in a training fold raises ValueError."""
        n = 40
        X = np.random.default_rng(0).normal(size=(n, 2))
        T = np.ones(n)
        Y = np.random.default_rng(0).normal(size=n)
        fold_spec = CrossFitSplit(n_folds=2, stratify=False).split(T, random_state=0)
        prop_model, out_model = _fast_models()
        with pytest.raises(ValueError, match="no control units"):
            fit_nuisance(X, T, Y, fold_spec, propensity_model=prop_model, outcome_model=out_model)


# ---------------------------------------------------------------------------
# dr_pseudo_outcome
# ---------------------------------------------------------------------------


class TestDRPseudoOutcome:
    def test_shape(self):
        X, T, Y, e_true, mu1_true, mu0_true = _make_dgp(n=100)
        nu = _oracle_nuisance(X, T, Y, e_true, mu1_true, mu0_true)
        gamma = dr_pseudo_outcome(Y, T, nu)
        assert gamma.shape == (100,)

    def test_formula_manual(self):
        """Check the formula against a manual computation on small arrays."""
        Y = np.array([1.0, 0.0, 1.0, 0.0])
        T = np.array([1.0, 0.0, 1.0, 0.0])
        e = np.array([0.6, 0.4, 0.7, 0.3])
        mu1 = np.array([0.8, 0.9, 0.85, 0.75])
        mu0 = np.array([0.2, 0.3, 0.25, 0.15])
        nu = NuisanceEstimates(e_hat=e, mu1_hat=mu1, mu0_hat=mu0)
        gamma = dr_pseudo_outcome(Y, T, nu)

        expected = (
            (mu1 - mu0)
            + T * (Y - mu1) / e
            - (1 - T) * (Y - mu0) / (1 - e)
        )
        np.testing.assert_allclose(gamma, expected)

    def test_oracle_unbiased(self):
        """With oracle nuisance, mean(DR pseudo-outcome) ≈ mean(true CATE)."""
        rng = np.random.default_rng(42)
        n = 2000
        X = rng.normal(size=(n, 2))
        e_true = 0.5 * np.ones(n)  # balanced RCT for clean test
        T = rng.binomial(1, e_true).astype(float)
        tau_true = X[:, 0] + X[:, 1]   # true CATE
        mu1_true = X[:, 0] + X[:, 1]
        mu0_true = np.zeros(n)
        Y = np.where(T == 1, mu1_true, mu0_true) + rng.normal(scale=0.5, size=n)

        nu = _oracle_nuisance(X, T, Y, e_true, mu1_true, mu0_true)
        gamma = dr_pseudo_outcome(Y, T, nu)

        # E[Gamma_i | X_i] = tau*(X_i); with n=2000 the empirical means should be close
        mean_gamma = gamma.mean()
        mean_tau = tau_true.mean()
        assert abs(mean_gamma - mean_tau) < 0.15, (
            f"DR pseudo-outcome mean {mean_gamma:.3f} far from true CATE mean {mean_tau:.3f}"
        )

    def test_pure_imputation_when_outcome_perfect(self):
        """With perfect outcome models, IPW corrections should vanish (in expectation)."""
        rng = np.random.default_rng(7)
        n = 500
        # Perfect outcome models: mu1_hat = mu1_true, mu0_hat = mu0_true
        X = rng.normal(size=(n, 2))
        e_true = np.full(n, 0.5)
        T = rng.binomial(1, e_true).astype(float)
        mu1_true = 2.0 * X[:, 0]
        mu0_true = X[:, 0]
        Y = np.where(T == 1, mu1_true, mu0_true) + rng.normal(scale=0.01, size=n)

        nu = _oracle_nuisance(X, T, Y, e_true, mu1_true, mu0_true)
        gamma = dr_pseudo_outcome(Y, T, nu)
        imputation = mu1_true - mu0_true  # tau*(x) = x[:, 0]

        # With near-perfect outcome models and small noise, gamma ≈ imputation
        np.testing.assert_allclose(gamma, imputation, atol=0.2)


# ---------------------------------------------------------------------------
# robinson_residuals
# ---------------------------------------------------------------------------


class TestRobinsonResiduals:
    def test_shapes(self):
        X, T, Y, e_true, mu1_true, mu0_true = _make_dgp(n=100)
        nu = _oracle_nuisance(X, T, Y, e_true, mu1_true, mu0_true)
        Y_tilde, W_tilde = robinson_residuals(Y, T, nu)
        assert Y_tilde.shape == (100,)
        assert W_tilde.shape == (100,)

    def test_formula_manual(self):
        Y = np.array([1.0, 2.0, 3.0])
        T = np.array([1.0, 0.0, 1.0])
        e = np.array([0.4, 0.6, 0.5])
        mu1 = np.array([1.1, 1.9, 2.8])
        mu0 = np.array([0.5, 0.9, 1.5])
        nu = NuisanceEstimates(e_hat=e, mu1_hat=mu1, mu0_hat=mu0)
        Y_tilde, W_tilde = robinson_residuals(Y, T, nu)

        m_hat = e * mu1 + (1 - e) * mu0
        np.testing.assert_allclose(Y_tilde, Y - m_hat)
        np.testing.assert_allclose(W_tilde, T - e)

    def test_partial_linear_structure(self):
        """With oracle nuisance in an RCT: Y_tilde ≈ W_tilde * tau + eps."""
        rng = np.random.default_rng(0)
        n = 1000
        X = rng.normal(size=(n, 2))
        e_true = np.full(n, 0.5)
        T = rng.binomial(1, e_true).astype(float)
        tau_true = X[:, 0]
        mu1_true = tau_true
        mu0_true = np.zeros(n)
        Y = np.where(T == 1, mu1_true, mu0_true) + rng.normal(scale=0.1, size=n)

        nu = _oracle_nuisance(X, T, Y, e_true, mu1_true, mu0_true)
        Y_tilde, W_tilde = robinson_residuals(Y, T, nu)

        # Y_tilde = W_tilde * tau(X) + residual; regression coefficient ≈ 1 for unit-scale tau
        # Check correlation between Y_tilde and W_tilde * tau_true
        lhs = Y_tilde
        rhs = W_tilde * tau_true
        corr = np.corrcoef(lhs, rhs)[0, 1]
        assert corr > 0.8, f"Partial linear structure weak: corr={corr:.3f}"

    def test_w_tilde_mean_near_zero(self):
        """In an RCT with e=0.5, W_tilde should have mean ~0."""
        rng = np.random.default_rng(1)
        n = 500
        X = rng.normal(size=(n, 2))
        e_true = np.full(n, 0.5)
        T = rng.binomial(1, e_true).astype(float)
        Y = rng.normal(size=n)
        nu = NuisanceEstimates(
            e_hat=np.full(n, 0.5),
            mu1_hat=np.zeros(n),
            mu0_hat=np.zeros(n),
        )
        _, W_tilde = robinson_residuals(Y, T, nu)
        assert abs(W_tilde.mean()) < 0.1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """fit_nuisance is deterministic given the same random_state."""

    def _hgb_models(self):
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )
        return (
            HistGradientBoostingClassifier(max_iter=20, early_stopping=True,
                                           validation_fraction=0.15),
            HistGradientBoostingRegressor(max_iter=20, early_stopping=True,
                                          validation_fraction=0.15),
        )

    def test_same_seed_identical_output(self):
        X, T, Y, *_ = _make_dgp(n=200, seed=7)
        fold_spec = CrossFitSplit(n_folds=3).split(T, random_state=0)

        nu1 = fit_nuisance(X, T, Y, fold_spec, *self._hgb_models(), random_state=42)
        nu2 = fit_nuisance(X, T, Y, fold_spec, *self._hgb_models(), random_state=42)

        np.testing.assert_array_equal(nu1.e_hat,   nu2.e_hat)
        np.testing.assert_array_equal(nu1.mu1_hat, nu2.mu1_hat)
        np.testing.assert_array_equal(nu1.mu0_hat, nu2.mu0_hat)

    def test_different_seeds_different_output(self):
        X, T, Y, *_ = _make_dgp(n=200, seed=7)
        fold_spec = CrossFitSplit(n_folds=3).split(T, random_state=0)

        nu42 = fit_nuisance(X, T, Y, fold_spec, *self._hgb_models(), random_state=42)
        nu99 = fit_nuisance(X, T, Y, fold_spec, *self._hgb_models(), random_state=99)

        assert not np.array_equal(nu42.e_hat, nu99.e_hat)


# ---------------------------------------------------------------------------
# Import from package
# ---------------------------------------------------------------------------


def test_importable_from_aggregation_package():
    from metacausal.aggregation import (  # noqa: F401
        NuisanceEstimates,
        dr_pseudo_outcome,
        fit_nuisance,
        robinson_residuals,
    )


def test_default_factories_importable():
    from metacausal.defaults import default_outcome_model, default_propensity_model  # noqa: F401
