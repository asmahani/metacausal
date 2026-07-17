"""Tests for CausalMLAdapter — meta-learners, tree-based, and TMLE."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal import CausalEnsemble, CausalMLAdapter
from metacausal.adapters.causalml import _is_causalml, _is_tmle, _is_uplift
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate

# CausalML imports
from causalml.inference.meta import (
    BaseDRRegressor,
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
    TMLELearner,
)
from causalml.inference.tree import (
    CausalTreeRegressor,
    UpliftRandomForestClassifier,
    UpliftTreeClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dgp(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n)
    Y_cont = X[:, 0] + T * X[:, 1] + rng.normal(scale=0.1, size=n)
    Y_bin = rng.binomial(1, 0.5, size=n)
    return X, T, Y_cont, Y_bin


def _fast_hgb_r():
    return HistGradientBoostingRegressor(max_iter=20)


def _fast_hgb_c():
    return HistGradientBoostingClassifier(max_iter=20)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


class TestDetectionHelpers:
    def test_is_causalml_meta(self):
        assert _is_causalml(BaseTRegressor(_fast_hgb_r()))

    def test_is_causalml_tree(self):
        assert _is_causalml(CausalTreeRegressor())

    def test_is_causalml_uplift(self):
        assert _is_causalml(UpliftTreeClassifier(control_name="control"))

    def test_is_causalml_tmle(self):
        assert _is_causalml(TMLELearner(_fast_hgb_r()))

    def test_is_not_causalml_sklearn(self):
        assert not _is_causalml(LogisticRegression())

    def test_is_tmle(self):
        assert _is_tmle(TMLELearner(_fast_hgb_r()))
        assert not _is_tmle(BaseTRegressor(_fast_hgb_r()))

    def test_is_uplift(self):
        assert _is_uplift(UpliftTreeClassifier(control_name="c"))
        assert not _is_uplift(BaseTRegressor(_fast_hgb_r()))


# ---------------------------------------------------------------------------
# Auto-detection in CausalEnsemble._wrap()
# ---------------------------------------------------------------------------


class TestAutoWrap:
    def test_causalml_estimator_auto_wrapped(self):
        X, T, Y_cont, _ = _dgp(n=200)
        ens = CausalEnsemble(
            methods=[BaseTRegressor(_fast_hgb_r())],  # not pre-wrapped
            aggregation="median",
        )
        ens.fit(X, T, Y_cont)
        adapter = ens._fitted_adapters[0]
        assert isinstance(adapter, CausalMLAdapter)

    def test_pre_wrapped_passed_through(self):
        X, T, Y_cont, _ = _dgp(n=200)
        wrapped = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()), name="my_t")
        ens = CausalEnsemble(methods=[wrapped], aggregation="median")
        ens.fit(X, T, Y_cont)
        assert ens._fitted_adapters[0].name == "my_t"


# ---------------------------------------------------------------------------
# Meta-learners
# ---------------------------------------------------------------------------


class TestMetaLearners:
    @pytest.mark.parametrize("cls", [BaseSRegressor, BaseTRegressor])
    def test_supports_cate(self, cls):
        a = CausalMLAdapter(cls(_fast_hgb_r()))
        assert a.supports_cate is True

    @pytest.mark.parametrize("cls", [BaseSRegressor, BaseTRegressor, BaseXRegressor])
    def test_cate_shape(self, cls):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(cls(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        result = a.cate(X)
        assert isinstance(result, ComponentCateEstimate)
        assert result.cate.shape == (200,)

    def test_cate_on_new_x(self):
        X, T, Y_cont, _ = _dgp(n=200)
        X_new = np.random.default_rng(99).normal(size=(50, 3))
        a = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        result = a.cate(X_new)
        assert result.cate.shape == (50,)

    def test_ate_with_ci(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        result = a.ate()
        assert isinstance(result, ComponentAteEstimate)
        assert np.isfinite(result.ate)
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.ci_upper

    def test_ate_ci_level_comes_from_wrapped_estimator(self):
        X, T, Y_cont, _ = _dgp(n=250, seed=7)

        a95 = CausalMLAdapter(BaseTRegressor(_fast_hgb_r(), ate_alpha=0.05))
        a95.fit(X, T, Y_cont, random_state=42)
        r95 = a95.ate()

        a90 = CausalMLAdapter(BaseTRegressor(_fast_hgb_r(), ate_alpha=0.10))
        a90.fit(X, T, Y_cont, random_state=42)
        r90 = a90.ate()

        width95 = r95.ci_upper - r95.ci_lower
        width90 = r90.ci_upper - r90.ci_lower
        assert width90 < width95

    @pytest.mark.parametrize("cls", [BaseRRegressor, BaseTRegressor])
    def test_ate_reproducible_across_calls(self, cls):
        """Regression for #16: two consecutive adapter.ate() calls on a
        fixed fit must be bit-identical. Prior to passing pretrain=True
        to causalml's estimate_ate(), the default pretrain=False silently
        refit the whole pipeline per call, making results stochastic."""
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(cls(_fast_hgb_r()))
        a.fit(X, T, Y_cont, random_state=42)
        r1 = a.ate()
        r2 = a.ate()
        assert r1.ate == r2.ate
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_name_defaults_to_class_name(self):
        a = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()))
        assert a.name == "BaseTRegressor"

    def test_name_override(self):
        a = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()), name="my_tlearner")
        assert a.name == "my_tlearner"

    def test_not_fitted_raises(self):
        a = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()))
        with pytest.raises(RuntimeError, match="not fitted"):
            a.cate(np.zeros((5, 3)))

    def test_deepcopy_before_fit(self):
        a = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()))
        a_copy = copy.deepcopy(a)
        assert not a_copy._is_fitted

    def test_deepcopy_after_fit(self):
        X, T, Y_cont, _ = _dgp(n=100)
        a = CausalMLAdapter(BaseTRegressor(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        a_copy = copy.deepcopy(a)
        np.testing.assert_array_equal(a.cate(X).cate, a_copy.cate(X).cate)

    def test_dr_regressor_seed_forwarded(self):
        """BaseDRRegressor accepts seed in fit() — verify it's forwarded."""
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(BaseDRRegressor(_fast_hgb_r(), _fast_hgb_r()))
        # Should not raise — seed param exists in BaseDRRegressor.fit()
        a.fit(X, T, Y_cont, random_state=42)
        assert a._is_fitted

    @pytest.mark.parametrize("make_model", [
        lambda: BaseSRegressor(_fast_hgb_r()),
        lambda: BaseTRegressor(_fast_hgb_r()),
        lambda: BaseRRegressor(
            outcome_learner=_fast_hgb_r(),
            propensity_learner=_fast_hgb_c(),
            effect_learner=_fast_hgb_r(),
        ),
        lambda: BaseDRRegressor(_fast_hgb_r(), _fast_hgb_r()),
    ], ids=["SRegressor", "TRegressor", "RRegressor", "DRRegressor"])
    def test_fit_reproducible_across_fresh_instances(self, make_model):
        """Regression for #16: two fresh adapter instances wrapping the
        same learner type, fit with the same random_state on identical
        data, must produce bit-identical predictions.

        Each causalml meta-learner stores its nested sklearn estimators
        under different internal names (``model``, ``model_c``/``model_t``,
        ``model_mu``/``model_tau``/``model_p``, ``model_mu_c``/
        ``model_mu_t``/``model_tau``), and BaseRRegressor additionally
        caches a ``KFold`` splitter at construction. Before #16 the
        adapter's ``_seed_model`` walked a hardcoded list of parameter
        names that matched none of these, so the seeds never propagated.
        """
        X, T, Y_cont, _ = _dgp(n=200)
        a1 = CausalMLAdapter(make_model())
        a1.fit(X, T, Y_cont, random_state=42)
        p1 = np.asarray(a1.cate(X).cate, dtype=float)

        a2 = CausalMLAdapter(make_model())
        a2.fit(X, T, Y_cont, random_state=42)
        p2 = np.asarray(a2.cate(X).cate, dtype=float)

        np.testing.assert_array_equal(p1, p2)

    def test_ensemble_with_meta_learner(self):
        X, T, Y_cont, _ = _dgp(n=200)
        ens = CausalEnsemble(
            methods=[CausalMLAdapter(BaseTRegressor(_fast_hgb_r()))],
            aggregation="median",
        )
        ens.fit(X, T, Y_cont)
        result = ens.cate(X)
        assert result.cate.shape == (200,)


# ---------------------------------------------------------------------------
# CausalTreeRegressor
# ---------------------------------------------------------------------------


class TestCausalTreeRegressor:
    def test_supports_cate(self):
        assert CausalMLAdapter(CausalTreeRegressor()).supports_cate is True

    def test_cate_shape(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(CausalTreeRegressor(min_samples_leaf=20))
        a.fit(X, T, Y_cont)
        result = a.cate(X)
        assert result.cate.shape == (200,)

    def test_ate_finite(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(CausalTreeRegressor(min_samples_leaf=20))
        a.fit(X, T, Y_cont)
        result = a.ate()
        assert np.isfinite(result.ate)

    def test_ate_has_ci(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(CausalTreeRegressor(min_samples_leaf=20))
        a.fit(X, T, Y_cont)
        result = a.ate()
        # CausalTreeRegressor.estimate_ate returns (ate, lb, ub)
        assert result.ci_lower is not None
        assert result.ci_lower <= result.ci_upper


# ---------------------------------------------------------------------------
# Uplift tree / forest
# ---------------------------------------------------------------------------


class TestUpliftTree:
    def test_supports_cate(self):
        a = CausalMLAdapter(
            UpliftTreeClassifier(control_name="control", max_depth=3,
                                 min_samples_leaf=20, min_samples_treatment=10)
        )
        assert a.supports_cate is True

    def test_cate_shape(self):
        X, T, _, Y_bin = _dgp(n=300)
        a = CausalMLAdapter(
            UpliftTreeClassifier(control_name="control", max_depth=3,
                                 min_samples_leaf=20, min_samples_treatment=10)
        )
        a.fit(X, T, Y_bin)
        result = a.cate(X)
        assert result.cate.shape == (300,)

    def test_cate_values_are_uplift(self):
        """CATE values are P(Y=1|T=1,X) - P(Y=1|T=0,X), range roughly [-1, 1]."""
        X, T, _, Y_bin = _dgp(n=300)
        a = CausalMLAdapter(
            UpliftTreeClassifier(control_name="control", max_depth=3,
                                 min_samples_leaf=20, min_samples_treatment=10)
        )
        a.fit(X, T, Y_bin)
        cate = a.cate(X).cate
        assert np.all(cate >= -1.0 - 1e-6)
        assert np.all(cate <= 1.0 + 1e-6)

    def test_string_treatment_conversion(self):
        """Binary T is internally converted to strings; user passes 0/1."""
        X, T, _, Y_bin = _dgp(n=300)
        a = CausalMLAdapter(
            UpliftTreeClassifier(control_name="control", max_depth=3,
                                 min_samples_leaf=20, min_samples_treatment=10)
        )
        # Should not raise — adapter handles the conversion
        a.fit(X, T, Y_bin)
        assert a._is_fitted

    def test_ate_fallback_to_mean_cate(self):
        """Uplift tree has no estimate_ate — should fall back to mean(cate)."""
        X, T, _, Y_bin = _dgp(n=300)
        a = CausalMLAdapter(
            UpliftTreeClassifier(control_name="control", max_depth=3,
                                 min_samples_leaf=20, min_samples_treatment=10)
        )
        a.fit(X, T, Y_bin)
        ate_result = a.ate()
        cate_result = a.cate(X)
        assert ate_result.ate == pytest.approx(cate_result.cate.mean())

    def test_uplift_forest(self):
        X, T, _, Y_bin = _dgp(n=300)
        a = CausalMLAdapter(
            UpliftRandomForestClassifier(
                control_name="control", n_estimators=5,
                min_samples_leaf=20, min_samples_treatment=10
            )
        )
        a.fit(X, T, Y_bin)
        result = a.cate(X)
        assert result.cate.shape == (300,)


# ---------------------------------------------------------------------------
# TMLE
# ---------------------------------------------------------------------------


class TestTMLE:
    def test_supports_cate_false(self):
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        assert a.supports_cate is False

    def test_cate_raises(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        with pytest.raises(NotImplementedError, match="ATE-only"):
            a.cate(X)

    def test_ate_returns_scalar(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        result = a.ate()
        assert isinstance(result, ComponentAteEstimate)
        assert np.isfinite(result.ate)

    def test_ate_has_ci(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        result = a.ate()
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.ci_upper

    def test_custom_propensity_model(self):
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(
            TMLELearner(_fast_hgb_r()),
            propensity_model=LogisticRegression(max_iter=200),
        )
        a.fit(X, T, Y_cont)
        assert a._p_hat is not None
        assert a._p_hat.shape == (200,)
        assert np.all(a._p_hat > 0) and np.all(a._p_hat < 1)

    def test_default_propensity_model_logistic(self):
        """Default propensity model is LogisticRegression."""
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        assert a._p_hat is not None

    def test_x_arg_ignored_for_tmle_ate(self):
        """For TMLE, ate() uses training data regardless of X argument."""
        X, T, Y_cont, _ = _dgp(n=200)
        X_other = np.random.default_rng(7).normal(size=(50, 3))
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        a.fit(X, T, Y_cont)
        ate_none = a.ate(None)
        ate_x = a.ate(X)
        ate_other = a.ate(X_other)  # should return same as training-data ATE
        assert ate_none.ate == pytest.approx(ate_x.ate)
        assert ate_x.ate == pytest.approx(ate_other.ate)

    def test_ate_reproducible_across_calls(self):
        """Regression for #16: TMLE's estimate_ate refits its outcome
        learner per call with an un-seeded val split, so without caching
        successive ate() calls drift. The adapter runs estimate_ate once
        at fit time and caches the triple."""
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        a.fit(X, T, Y_cont, random_state=42)
        r1 = a.ate()
        r2 = a.ate()
        assert r1.ate == r2.ate
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_ate_computed_at_fit_time(self):
        """The cached ATE triple is populated during fit(), not on the
        first ate() call — so a crashed adapter.ate() call won't run
        TMLE under the hood."""
        X, T, Y_cont, _ = _dgp(n=200)
        a = CausalMLAdapter(TMLELearner(_fast_hgb_r()))
        a.fit(X, T, Y_cont, random_state=42)
        assert a._tmle_ate_cache is not None
        cached_ate, cached_lb, cached_ub = a._tmle_ate_cache
        assert np.isfinite(cached_ate)
        assert cached_lb <= cached_ate <= cached_ub

    def test_tmle_in_pointwise_ensemble(self):
        """TMLE contributes ATE to a pointwise (median) ensemble."""
        X, T, Y_cont, _ = _dgp(n=200)
        ens = CausalEnsemble(
            methods=[
                CausalMLAdapter(BaseTRegressor(_fast_hgb_r())),
                CausalMLAdapter(TMLELearner(_fast_hgb_r())),
            ],
            aggregation="median",
        )
        ens.fit(X, T, Y_cont)
        result = ens.ate(X)
        assert np.isfinite(result.ate)

    def test_tmle_excluded_from_cate_ensemble(self):
        """TMLE (supports_cate=False) is excluded from cate() computation."""
        X, T, Y_cont, _ = _dgp(n=200)
        ens = CausalEnsemble(
            methods=[
                CausalMLAdapter(BaseTRegressor(_fast_hgb_r()), name="tlearner"),
                CausalMLAdapter(TMLELearner(_fast_hgb_r()), name="tmle"),
            ],
            aggregation="median",
        )
        ens.fit(X, T, Y_cont)
        result = ens.cate(X)
        # TMLE excluded from CATE
        assert "tmle" not in result.component_estimates
        assert "tlearner" in result.component_estimates


class TestPropensityThreading:
    """``CausalMLAdapter`` pre-fits a wired-in ``propensity_model`` and
    threads it through ``model.fit(..., p=...)`` for any non-TMLE
    CausalML estimator whose fit signature accepts ``p``.

    Without this, ``BaseXLearner`` family silently falls through to
    CausalML's ``ElasticNetPropensityModel()`` (LogisticRegressionCV
    with saga), which (a) ignores the user's nuisance choice and
    (b) floods ``ConvergenceWarning`` on small data.
    """

    def test_x_learner_receives_propensity(self):
        """When ``propensity_model`` is supplied, the adapter pre-fits
        it and threads the scores through ``BaseXRegressor.fit(p=...)``.
        Verified by a spy subclass that captures the ``p`` kwarg
        (deepcopy preserves the class, so the spy still fires after the
        adapter's ``copy.deepcopy(model)``)."""
        X, T, Y_cont, _ = _dgp(n=200)
        captured: dict = {"p": "NOT-PASSED"}

        class SpyXRegressor(BaseXRegressor):
            def fit(self, X, treatment, y, p=None, **kw):
                captured["p"] = p
                return super().fit(X, treatment, y, p=p, **kw)

        a = CausalMLAdapter(
            SpyXRegressor(_fast_hgb_r()),
            propensity_model=LogisticRegression(max_iter=200),
        )
        a.fit(X, T, Y_cont)

        p_passed = captured["p"]
        assert p_passed is not None and not isinstance(p_passed, str)
        p_arr = np.asarray(p_passed)
        assert p_arr.shape == (200,)
        assert np.all((p_arr > 0) & (p_arr < 1))

    def test_no_propensity_model_no_threading(self):
        """Without ``propensity_model``, the adapter does not synthesize
        one, leaving CausalML to fall through to its internal default."""
        X, T, Y_cont, _ = _dgp(n=100)
        x_learner = BaseXRegressor(_fast_hgb_r())
        a = CausalMLAdapter(x_learner)
        # Should not raise; the propensity falls through to CausalML's
        # internal default (we don't assert anything about that — just
        # that the adapter doesn't crash and a fitted model is produced).
        a.fit(X, T, Y_cont)
        assert a._fitted_model is not None
        assert a._is_fitted

    def test_propensity_only_passed_when_signature_accepts_it(self):
        """When the wrapped model's fit signature has no ``p`` parameter
        (e.g. ``BaseSRegressor``), the adapter must not pass ``p=`` even
        if a propensity_model is supplied — that would TypeError."""
        X, T, Y_cont, _ = _dgp(n=100)
        s_learner = BaseSRegressor(_fast_hgb_r())
        a = CausalMLAdapter(
            s_learner,
            propensity_model=LogisticRegression(max_iter=200),
        )
        # Must not raise (signature gating prevents the TypeError).
        a.fit(X, T, Y_cont)
        assert a._is_fitted

    def test_propensity_threaded_into_cate_predict(self):
        """When fit pre-fits the propensity model, ``cate(X_new)`` must
        also pass ``p=`` to the wrapped predict — otherwise
        BaseXLearner.predict() falls back to ``self.propensity_model``,
        which is unset because we passed ``p=`` at fit time. Regression
        guard for the binary defaults pool."""
        X, T, Y_cont, _ = _dgp(n=200)
        captured: dict = {"p": "NOT-PASSED"}

        class SpyXRegressor(BaseXRegressor):
            def predict(self, X, treatment=None, y=None, p=None, **kw):
                captured["p"] = p
                return super().predict(X, treatment=treatment, y=y, p=p, **kw)

        a = CausalMLAdapter(
            SpyXRegressor(_fast_hgb_r()),
            propensity_model=LogisticRegression(max_iter=200),
        )
        a.fit(X, T, Y_cont)

        X_new = np.random.default_rng(7).normal(size=(50, X.shape[1]))
        cate = a.cate(X_new)
        assert cate.cate.shape == (50,)

        p_passed = captured["p"]
        assert p_passed is not None and not isinstance(p_passed, str)
        p_arr = np.asarray(p_passed)
        assert p_arr.shape == (50,)
        assert np.all((p_arr > 0) & (p_arr < 1))

    def test_propensity_thread_does_not_set_p_hat(self):
        """``_p_hat`` is reserved for the TMLE branch; the X-learner
        propensity threading uses a local var and must not write to it."""
        X, T, Y_cont, _ = _dgp(n=100)
        a = CausalMLAdapter(
            BaseXRegressor(_fast_hgb_r()),
            propensity_model=LogisticRegression(max_iter=200),
        )
        a.fit(X, T, Y_cont)
        assert a._p_hat is None  # TMLE-only attribute, untouched here


# ---------------------------------------------------------------------------
# Nested-parallelism suppression (issue #47)
# ---------------------------------------------------------------------------


class TestForceSerialUnderParallelism:
    """Inside a MetaCausal worker (``INNER_WORKER_ENV`` set), the adapter
    pins the wrapped R-Learner's ``cv_n_jobs`` to 1 so its internal
    ``cross_val_predict`` pool cannot nest inside the outer worker. Outside
    a worker, ``cv_n_jobs`` is left untouched. Mirrors
    ``test_econml_adapter.py::TestForceSerialUnderParallelism``.
    """

    def test_force_serial_pins_cv_n_jobs(self):
        model = BaseRRegressor(outcome_learner=_fast_hgb_r(), effect_learner=_fast_hgb_r())
        assert model.cv_n_jobs == -1
        CausalMLAdapter._force_serial(model)
        assert model.cv_n_jobs == 1

    def test_fit_pins_cv_n_jobs_inside_worker(self, monkeypatch):
        from metacausal._parallel import INNER_WORKER_ENV

        monkeypatch.setenv(INNER_WORKER_ENV, "1")
        X, T, Y_cont, _ = _dgp(n=100, seed=8)
        template = BaseRRegressor(outcome_learner=_fast_hgb_r(), effect_learner=_fast_hgb_r())
        adapter = CausalMLAdapter(template)
        adapter.fit(X, T, Y_cont, random_state=8)
        assert adapter._fitted_model.cv_n_jobs == 1
        # The original template is left untouched.
        assert adapter._model.cv_n_jobs == -1

    def test_fit_leaves_cv_n_jobs_outside_worker(self, monkeypatch):
        from metacausal._parallel import INNER_WORKER_ENV

        monkeypatch.delenv(INNER_WORKER_ENV, raising=False)
        X, T, Y_cont, _ = _dgp(n=100, seed=9)
        template = BaseRRegressor(outcome_learner=_fast_hgb_r(), effect_learner=_fast_hgb_r())
        adapter = CausalMLAdapter(template)
        adapter.fit(X, T, Y_cont, random_state=9)
        assert adapter._fitted_model.cv_n_jobs == -1


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_importable_from_top_level():
    from metacausal import CausalMLAdapter  # noqa: F401


def test_importable_from_adapters():
    from metacausal.adapters import CausalMLAdapter  # noqa: F401
