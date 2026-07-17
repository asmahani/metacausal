"""Tests for StochtreeAdapter using a mocked BCFModel."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metacausal.adapters.stochtree import StochtreeAdapter


def _make_mock_bcf(n_obs: int, num_mcmc: int):
    """Return a mock BCFModel whose predict returns a known array."""
    mock_model = MagicMock()
    # tau_samples shape: (n_obs, num_mcmc)
    # Row i = unit i, column j = posterior draw j
    # Use distinct per-draw means so ATE draws are distinguishable
    rng = np.random.default_rng(0)
    tau = rng.normal(loc=2.0, scale=0.1, size=(n_obs, num_mcmc))
    mock_model.predict.return_value = tau
    return mock_model, tau


def _dummy_data(n=50, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    T = rng.binomial(1, 0.5, size=n)
    Y = X[:, 0] + T * 2.0 + rng.normal(size=n)
    return X, T, Y


class TestAxisFix:
    """Verify that ATE draws are per-posterior-sample, not per-unit."""

    @patch("stochtree.BCFModel")
    def test_ate_draws_length_equals_num_mcmc(self, MockBCF):
        n_obs, num_mcmc = 50, 20
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_gfr=0, num_burnin=0, num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)
        ps = np.full(n_obs, 0.5)
        adapter.fit(X, T, Y, propensity=ps, random_state=42)
        result = adapter.ate()

        ate_draws = result.details["ate_draws"]
        assert ate_draws.shape == (num_mcmc,)

    @patch("stochtree.BCFModel")
    def test_ci_reflects_posterior_not_heterogeneity(self, MockBCF):
        """CIs should narrow when posterior draws agree, even if units differ."""
        n_obs, num_mcmc = 100, 50
        mock_model = MagicMock()
        # All posterior draws have the same ATE (no posterior uncertainty)
        # but units differ wildly within each draw
        tau = np.random.default_rng(0).normal(
            loc=2.0, scale=5.0, size=(n_obs, num_mcmc)
        )
        # Force each column (draw) to have the same mean
        tau = tau - tau.mean(axis=0, keepdims=True) + 2.0
        mock_model.predict.return_value = tau
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)
        ps = np.full(n_obs, 0.5)
        adapter.fit(X, T, Y, propensity=ps, random_state=1)
        result = adapter.ate()

        # With no posterior uncertainty, CI should be very tight
        assert result.ci_upper - result.ci_lower < 0.01


class TestParamPassthrough:
    """Verify that dicts are forwarded to BCFModel.sample()."""

    @patch("stochtree.BCFModel")
    def test_default_dicts_empty(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_mcmc=10)
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5), random_state=1)

        call_kwargs = mock_model.sample.call_args[1]
        assert call_kwargs["prognostic_forest_params"] == {}
        assert call_kwargs["treatment_effect_forest_params"] == {}

    @patch("stochtree.BCFModel")
    def test_custom_forest_params_forwarded(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        prog = {"num_trees": 200, "alpha": 0.9}
        trt = {"num_trees": 30, "max_depth": 3}
        adapter = StochtreeAdapter(
            num_mcmc=10,
            prognostic_forest_params=prog,
            treatment_effect_forest_params=trt,
        )
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5), random_state=1)

        call_kwargs = mock_model.sample.call_args[1]
        assert call_kwargs["prognostic_forest_params"] == prog
        assert call_kwargs["treatment_effect_forest_params"] == trt

    @patch("stochtree.BCFModel")
    def test_general_params_forwarded(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        gp = {"propensity_covariate": "both", "standardize": False}
        adapter = StochtreeAdapter(num_mcmc=10, general_params=gp)
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5), random_state=7)

        call_kwargs = mock_model.sample.call_args[1]
        passed_gp = call_kwargs["general_params"]
        assert passed_gp["propensity_covariate"] == "both"
        assert passed_gp["standardize"] is False
        # Ensemble seed should also be injected
        assert "random_seed" in passed_gp


class TestSeedMerging:
    """Verify random_seed precedence and propensity seed derivation."""

    @patch("stochtree.BCFModel")
    def test_ensemble_seed_injected(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_mcmc=10)
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5), random_state=99)

        gp = mock_model.sample.call_args[1]["general_params"]
        assert gp["random_seed"] == 99

    @patch("stochtree.BCFModel")
    def test_ensemble_seed_overrides_user_seed(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(
            num_mcmc=10, general_params={"random_seed": 5}
        )
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5), random_state=99)

        gp = mock_model.sample.call_args[1]["general_params"]
        assert gp["random_seed"] == 99

    @patch("stochtree.BCFModel")
    def test_user_seed_used_when_no_ensemble_seed(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(
            num_mcmc=10, general_params={"random_seed": 42}
        )
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5))

        gp = mock_model.sample.call_args[1]["general_params"]
        assert gp["random_seed"] == 42

    @patch("stochtree.BCFModel")
    def test_no_seed_when_none_provided(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_mcmc=10)
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5))

        gp = mock_model.sample.call_args[1]["general_params"]
        assert "random_seed" not in gp

    @patch("stochtree.BCFModel")
    def test_propensity_seed_differs_from_bcf_seed(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_mcmc=10)
        X, T, Y = _dummy_data()
        # Don't pass propensity — force propensity computation
        with patch.object(adapter, "_compute_propensity", return_value=np.full(50, 0.5)) as mock_prop:
            adapter.fit(X, T, Y, random_state=99)
            prop_seed = mock_prop.call_args[0][2]
            bcf_seed = mock_model.sample.call_args[1]["general_params"]["random_seed"]
            assert prop_seed != bcf_seed
            assert prop_seed == (bcf_seed + 1) % (2**31)


class TestUserGeneralParamsNotMutated:
    """Ensure the user's original dict is not modified."""

    @patch("stochtree.BCFModel")
    def test_original_dict_unchanged(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        original_gp = {"standardize": False}
        adapter = StochtreeAdapter(num_mcmc=10, general_params=original_gp)
        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5), random_state=42)

        # The adapter should have copied the dict, not mutated it
        assert "random_seed" not in original_gp


class TestFitPredict:
    """Verify fit/predict separation for StochtreeAdapter."""

    @patch("stochtree.BCFModel")
    def test_is_fitted_flag(self, MockBCF):
        mock_model, _ = _make_mock_bcf(50, 10)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_mcmc=10)
        assert not adapter._is_fitted

        X, T, Y = _dummy_data()
        adapter.fit(X, T, Y, propensity=np.full(50, 0.5), random_state=1)
        assert adapter._is_fitted

    @patch("stochtree.BCFModel")
    def test_ate_before_fit_raises(self, MockBCF):
        adapter = StochtreeAdapter(num_mcmc=10)
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.ate()

    @patch("stochtree.BCFModel")
    def test_supports_cate(self, MockBCF):
        adapter = StochtreeAdapter()
        assert adapter.supports_cate is True

    @patch("stochtree.BCFModel")
    def test_cate_before_fit_raises(self, MockBCF):
        adapter = StochtreeAdapter(num_mcmc=10)
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.cate(np.zeros((10, 3)))


class TestCate:
    """Verify CATE extraction from stochtree posterior."""

    @patch("stochtree.BCFModel")
    def test_cate_shape(self, MockBCF):
        n_obs, num_mcmc = 50, 20
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_gfr=0, num_burnin=0, num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)
        adapter.fit(X, T, Y, propensity=np.full(n_obs, 0.5), random_state=42)
        result = adapter.cate(X)

        assert result.cate.shape == (n_obs,)
        assert result.ci_lower.shape == (n_obs,)
        assert result.ci_upper.shape == (n_obs,)

    @patch("stochtree.BCFModel")
    def test_cate_values_are_posterior_means(self, MockBCF):
        """CATE point estimates should be posterior mean per observation."""
        n_obs, num_mcmc = 30, 15
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_gfr=0, num_burnin=0, num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)
        adapter.fit(X, T, Y, propensity=np.full(n_obs, 0.5), random_state=42)
        result = adapter.cate(X)

        expected_cate = tau.mean(axis=1)
        np.testing.assert_allclose(result.cate, expected_cate)

    @patch("stochtree.BCFModel")
    def test_cate_ci_are_quantiles(self, MockBCF):
        """CATE CIs should be posterior quantiles per observation."""
        n_obs, num_mcmc = 40, 100
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        MockBCF.return_value = mock_model

        alpha = 0.1
        adapter = StochtreeAdapter(
            num_gfr=0, num_burnin=0, num_mcmc=num_mcmc, alpha=alpha
        )
        X, T, Y = _dummy_data(n=n_obs)
        adapter.fit(X, T, Y, propensity=np.full(n_obs, 0.5), random_state=42)
        result = adapter.cate(X)

        expected_lo = np.quantile(tau, alpha / 2, axis=1)
        expected_hi = np.quantile(tau, 1 - alpha / 2, axis=1)
        np.testing.assert_allclose(result.ci_lower, expected_lo)
        np.testing.assert_allclose(result.ci_upper, expected_hi)


class TestOutOfSamplePropensity:
    """Verify propensity estimation for out-of-sample predictions."""

    @patch("stochtree.BCFModel")
    def test_out_of_sample_uses_fitted_propensity_models(self, MockBCF):
        """When propensity was computed during fit, out-of-sample predict
        should use the stored CV models, not dummy 0.5."""
        n_obs, num_mcmc = 50, 10
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_gfr=0, num_burnin=0, num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)

        # Fit with computed propensity (not user-provided)
        with patch.object(
            adapter, "_compute_propensity", return_value=np.full(n_obs, 0.5)
        ):
            adapter.fit(X, T, Y, random_state=42)

        # Inject mock propensity models
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.column_stack([
            np.full(20, 0.4), np.full(20, 0.6)
        ])
        adapter._fitted_propensity_models = [mock_clf, mock_clf]

        # Out-of-sample prediction on new X
        X_test = np.zeros((20, 3))
        # Mock predict to return different tau for new X
        tau_new = np.random.default_rng(0).normal(size=(20, num_mcmc))
        mock_model.predict.return_value = tau_new
        adapter.cate(X_test)

        # Verify predict was called with estimated propensity, not 0.5
        predict_call = mock_model.predict.call_args
        propensity_arg = predict_call[1]["propensity"]
        np.testing.assert_allclose(propensity_arg, 0.6)

    @patch("stochtree.BCFModel")
    def test_out_of_sample_warns_when_no_propensity_models(self, MockBCF):
        """When user provided propensity manually, out-of-sample falls back
        to 0.5 with a warning."""
        n_obs, num_mcmc = 50, 10
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_gfr=0, num_burnin=0, num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)

        # Fit with user-provided propensity (no CV models stored)
        adapter.fit(X, T, Y, propensity=np.full(n_obs, 0.3), random_state=42)
        assert adapter._fitted_propensity_models is None

        # Out-of-sample should warn and use 0.5
        X_test = np.zeros((20, 3))
        tau_new = np.random.default_rng(0).normal(size=(20, num_mcmc))
        mock_model.predict.return_value = tau_new

        with pytest.warns(RuntimeWarning, match="No fitted propensity models"):
            adapter.cate(X_test)

        predict_call = mock_model.predict.call_args
        propensity_arg = predict_call[1]["propensity"]
        np.testing.assert_allclose(propensity_arg, 0.5)

    @patch("stochtree.BCFModel")
    def test_out_of_sample_propensity_clipped(self, MockBCF):
        """Out-of-sample propensity should be clipped like training propensity."""
        n_obs, num_mcmc = 50, 10
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        MockBCF.return_value = mock_model

        clip_eps = 0.05
        adapter = StochtreeAdapter(
            num_gfr=0, num_burnin=0, num_mcmc=num_mcmc,
            propensity_clip_eps=clip_eps,
        )
        X, T, Y = _dummy_data(n=n_obs)

        with patch.object(
            adapter, "_compute_propensity", return_value=np.full(n_obs, 0.5)
        ):
            adapter.fit(X, T, Y, random_state=42)

        # Mock propensity model that returns extreme values
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.column_stack([
            np.full(10, 0.99), np.full(10, 0.01)  # extreme propensity
        ])
        adapter._fitted_propensity_models = [mock_clf]

        X_test = np.zeros((10, 3))
        tau_new = np.random.default_rng(0).normal(size=(10, num_mcmc))
        mock_model.predict.return_value = tau_new
        adapter.cate(X_test)

        predict_call = mock_model.predict.call_args
        propensity_arg = predict_call[1]["propensity"]
        assert propensity_arg.min() >= clip_eps
        assert propensity_arg.max() <= 1 - clip_eps


# ---------------------------------------------------------------------------
# Tests: pickle round-trip (__getstate__ / __setstate__)
# ---------------------------------------------------------------------------


class TestStochtreePickle:
    @patch("stochtree.BCFModel")
    def test_unfitted_adapter_picklable(self, MockBCF):
        """Unfitted adapter (no BCFModel) pickles and unpickles cleanly."""
        adapter = StochtreeAdapter()
        restored = pickle.loads(pickle.dumps(adapter))
        assert restored._fitted_model is None
        assert not restored._is_fitted

    @patch("stochtree.BCFModel")
    def test_fitted_adapter_round_trip(self, MockBCF):
        """Fitted adapter serializes BCFModel via to_json and restores via from_json."""
        n_obs, num_mcmc = 30, 5
        mock_model, tau = _make_mock_bcf(n_obs, num_mcmc)
        json_str = '{"mock": "json"}'
        mock_model.to_json.return_value = json_str
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_gfr=0, num_burnin=0, num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)
        adapter.fit(X, T, Y, propensity=np.full(n_obs, 0.5))

        # Pickle / unpickle
        with patch("stochtree.BCFModel") as MockBCF2:
            mock_model2 = MagicMock()
            MockBCF2.return_value = mock_model2
            restored = pickle.loads(pickle.dumps(adapter))

        # BCFModel() was constructed and from_json called with the JSON string
        MockBCF2.assert_called_once()
        mock_model2.from_json.assert_called_once_with(json_str)

        # Fitted state is preserved
        assert restored._is_fitted
        assert restored._fitted_model is mock_model2
        np.testing.assert_array_equal(restored._tau_samples, tau)

    @patch("stochtree.BCFModel")
    def test_getstate_removes_model_key(self, MockBCF):
        """__getstate__ stores JSON under _fitted_model_json, not _fitted_model."""
        n_obs, num_mcmc = 20, 5
        mock_model, _ = _make_mock_bcf(n_obs, num_mcmc)
        mock_model.to_json.return_value = '{"x": 1}'
        MockBCF.return_value = mock_model

        adapter = StochtreeAdapter(num_gfr=0, num_burnin=0, num_mcmc=num_mcmc)
        X, T, Y = _dummy_data(n=n_obs)
        adapter.fit(X, T, Y, propensity=np.full(n_obs, 0.5))

        state = adapter.__getstate__()
        assert "_fitted_model" not in state
        assert state["_fitted_model_json"] == '{"x": 1}'

    @patch("stochtree.BCFModel")
    def test_setstate_restores_none_when_not_fitted(self, MockBCF):
        """__setstate__ with no _fitted_model_json key leaves _fitted_model as None."""
        adapter = StochtreeAdapter()
        state = adapter.__getstate__()
        assert "_fitted_model_json" not in state

        fresh = StochtreeAdapter.__new__(StochtreeAdapter)
        fresh.__setstate__(state)
        assert fresh._fitted_model is None


def test_importable_from_top_level():
    from metacausal import StochtreeAdapter  # noqa: F401
