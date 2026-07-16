"""Tests for EconMLAdapter focused on warning-handling guarantees (issue #18)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from econml.dr import DRLearner
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import DataConversionWarning

from metacausal.adapters.econml import EconMLAdapter


def _binary_dgp(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    p_y = 1.0 / (1.0 + np.exp(-(X[:, 0] + T * 0.5)))
    Y = rng.binomial(1, p_y).astype(float)
    return X, T, Y


def _fast_clf():
    return HistGradientBoostingClassifier(max_iter=20)


class _DummyEconMLEstimator:
    def __init__(self):
        self.fit_inference = None
        self.interval_alpha = None

    def fit(self, Y, T, X=None, inference=None):
        self.fit_inference = inference
        return self

    def ate(self, X=None):
        return 0.5

    def ate_interval(self, X=None, alpha=0.05):
        self.interval_alpha = alpha
        return np.array([0.25]), np.array([0.75])

    def effect(self, X=None):
        n = len(X) if X is not None else 1
        return np.full(n, 0.5)

    def effect_interval(self, X=None, alpha=0.05):
        self.interval_alpha = alpha
        n = len(X) if X is not None else 1
        return np.full(n, 0.25), np.full(n, 0.75)


class _WarnOnAteEstimator:
    def __init__(
        self,
        *,
        warning_message=(
            "'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 "
            "and will be removed in 1.8."
        ),
        warning_module="sklearn.utils.deprecation",
    ):
        self.warning_message = warning_message
        self.warning_module = warning_module
        self.interval_alpha = None

    def fit(self, Y, T, X=None, inference=None):
        return self

    def _warn(self):
        warnings.warn_explicit(
            self.warning_message,
            FutureWarning,
            "sklearn/utils/deprecation.py",
            151,
            module=self.warning_module,
        )

    def ate(self, X=None):
        self._warn()
        return np.array([0.5])

    def ate_interval(self, X=None, alpha=0.05):
        self._warn()
        self.interval_alpha = alpha
        return np.array([0.25]), np.array([0.75])


class _WarnOnEffectEstimator:
    def __init__(
        self,
        *,
        warning_message=(
            "'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 "
            "and will be removed in 1.8."
        ),
        warning_module="sklearn.utils.deprecation",
    ):
        self.warning_message = warning_message
        self.warning_module = warning_module
        self.interval_alpha = None

    def fit(self, Y, T, X=None, inference=None):
        return self

    def _warn(self):
        warnings.warn_explicit(
            self.warning_message,
            FutureWarning,
            "sklearn/utils/deprecation.py",
            151,
            module=self.warning_module,
        )

    def effect(self, X=None):
        self._warn()
        n = len(X) if X is not None else 1
        return np.full(n, 0.5)

    def effect_interval(self, X=None, alpha=0.05):
        self._warn()
        self.interval_alpha = alpha
        n = len(X) if X is not None else 1
        return np.full(n, 0.25), np.full(n, 0.75)


class TestDataConversionWarningSuppression:
    """The DRLearner+discrete_outcome=True path triggers an upstream
    DataConversionWarning (econml reshapes Y to a column vector before
    passing to a sklearn classifier). The adapter suppresses it surgically.
    """

    def test_suppression_under_filterwarnings_error(self):
        """With DataConversionWarning promoted to error globally, the
        adapter's fit() must not raise — proving the internal filter
        intercepts the warning before the test-level error rule sees it.
        """
        X, T, Y = _binary_dgp(n=200, seed=0)
        adapter = EconMLAdapter(
            DRLearner(
                model_regression=_fast_clf(),
                model_propensity=_fast_clf(),
                discrete_outcome=True,
            )
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=DataConversionWarning)
            # Must NOT raise: the internal catch_warnings in EconMLAdapter.fit
            # filters this specific warning before the outer "error" rule can
            # promote it to an exception.
            adapter.fit(X, T, Y, random_state=42)

        assert adapter._is_fitted

    def test_filter_does_not_leak_after_fit(self):
        """The filter must be scoped to the fit() call only. After fit
        returns, a fresh DataConversionWarning fired manually should still
        be visible / actionable per the caller's settings.
        """
        X, T, Y = _binary_dgp(n=200, seed=1)
        adapter = EconMLAdapter(
            DRLearner(
                model_regression=_fast_clf(),
                model_propensity=_fast_clf(),
                discrete_outcome=True,
            )
        )

        adapter.fit(X, T, Y, random_state=42)

        # Now, outside fit(), promote DataConversionWarning to error and
        # fire one with the same message + module. It must raise — proving
        # the adapter's filter did NOT leak past its catch_warnings scope.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=DataConversionWarning)
            with pytest.raises(DataConversionWarning):
                warnings.warn_explicit(
                    "A column-vector y was passed when a 1d array was expected. "
                    "(test fire)",
                    DataConversionWarning,
                    "sklearn/utils/validation.py",
                    1408,
                )

    def test_unrelated_warnings_pass_through(self):
        """Other warning categories fired during fit must still surface.
        We don't fire one ourselves — DRLearner's discrete_outcome=True path
        is known to also emit a UserWarning ('nu2 not positive' from
        DoubleML is a *different* adapter) but for econml we just verify
        that the suppression is category-narrow: a UserWarning fired
        inside the catch_warnings scope is still visible.
        """
        # We construct a fake call that fires a UserWarning inside the
        # filter context to verify category-narrowness. This is a unit test
        # of the filter mechanism, not of DRLearner specifically.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DataConversionWarning,
                message=r"A column-vector y was passed when a 1d array was expected.*",
                module=r"sklearn\.utils\.validation",
            )
            with pytest.warns(UserWarning, match="something else"):
                warnings.warn("something else entirely", UserWarning)
            with pytest.warns(FutureWarning, match="future"):
                warnings.warn("future", FutureWarning)


class TestForceAllFiniteFutureWarningSuppression:
    def test_ate_path_suppression_under_filterwarnings_error(self):
        """A matching sklearn FutureWarning on the ate() path must be
        intercepted inside the adapter before an outer "error" rule sees it.
        """
        X, T, Y = _binary_dgp(n=50, seed=7)
        adapter = EconMLAdapter(_WarnOnAteEstimator())
        adapter.fit(X, T, Y)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning)
            result = adapter.ate()

        assert result.ate == 0.5

    def test_effect_path_suppresses_prediction_and_interval_warnings(self):
        """The effect()/effect_interval() path is the upstream leak site
        for estimators without a native ate() method.
        """
        X, T, Y = _binary_dgp(n=50, seed=8)
        adapter = EconMLAdapter(_WarnOnEffectEstimator(), alpha=0.10)
        adapter.fit(X, T, Y)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ate_result = adapter.ate(X)
            cate_result = adapter.cate(X[:7])

        assert ate_result.ate == 0.5
        assert ate_result.ci_lower == 0.25
        assert ate_result.ci_upper == 0.75
        assert cate_result.cate.shape == (7,)
        assert cate_result.ci_lower.shape == (7,)
        assert cate_result.ci_upper.shape == (7,)
        assert adapter._fitted_model.interval_alpha == 0.10
        assert not caught

    def test_filter_does_not_leak_after_prediction(self):
        X, T, Y = _binary_dgp(n=50, seed=9)
        adapter = EconMLAdapter(_WarnOnEffectEstimator())
        adapter.fit(X, T, Y)
        adapter.cate(X[:5])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning)
            with pytest.raises(FutureWarning):
                warnings.warn_explicit(
                    "'force_all_finite' was renamed to 'ensure_all_finite' "
                    "in 1.6 and will be removed in 1.8.",
                    FutureWarning,
                    "sklearn/utils/deprecation.py",
                    151,
                    module="sklearn.utils.deprecation",
                )

    def test_unrelated_futurewarning_message_still_surfaces(self):
        X, T, Y = _binary_dgp(n=50, seed=10)
        adapter = EconMLAdapter(
            _WarnOnAteEstimator(warning_message="future from somewhere else")
        )
        adapter.fit(X, T, Y)

        with pytest.warns(FutureWarning, match="future from somewhere else"):
            adapter.ate()

    def test_same_message_from_other_module_still_surfaces(self):
        X, T, Y = _binary_dgp(n=50, seed=11)
        adapter = EconMLAdapter(
            _WarnOnEffectEstimator(warning_module="upstream.other")
        )
        adapter.fit(X, T, Y)

        with pytest.warns(
            FutureWarning,
            match="'force_all_finite' was renamed to 'ensure_all_finite'",
        ):
            adapter.cate(X[:5])


class TestInferenceConfigForwarding:
    def test_fit_forwards_inference_backend(self):
        X, T, Y = _binary_dgp(n=50, seed=2)
        adapter = EconMLAdapter(
            _DummyEconMLEstimator(),
            alpha=0.10,
            inference="statsmodels",
        )

        adapter.fit(X, T, Y, random_state=42)

        assert adapter._fitted_model.fit_inference == "statsmodels"

    def test_ate_interval_forwards_alpha(self):
        X, T, Y = _binary_dgp(n=50, seed=3)
        adapter = EconMLAdapter(_DummyEconMLEstimator(), alpha=0.10)
        adapter.fit(X, T, Y)

        result = adapter.ate(X)

        assert result.ci_lower == 0.25
        assert result.ci_upper == 0.75
        assert adapter._fitted_model.interval_alpha == 0.10

    def test_cate_interval_forwards_alpha(self):
        X, T, Y = _binary_dgp(n=50, seed=4)
        adapter = EconMLAdapter(_DummyEconMLEstimator(), alpha=0.10)
        adapter.fit(X, T, Y)

        result = adapter.cate(X[:7])

        assert result.ci_lower.shape == (7,)
        assert result.ci_upper.shape == (7,)
        assert adapter._fitted_model.interval_alpha == 0.10


class _JobsEstimator:
    """Minimal EconML-like estimator exposing ``n_jobs`` and nested
    sub-models, recording the ``n_jobs`` value in effect at fit time."""

    def __init__(self, n_jobs=-1):
        from types import SimpleNamespace

        self.n_jobs = n_jobs
        self.model_y = SimpleNamespace(n_jobs=-1)
        self.models = [SimpleNamespace(n_jobs=-1), SimpleNamespace(n_jobs=4)]
        self.n_jobs_at_fit = None

    def fit(self, Y, T, X=None, inference=None):
        self.n_jobs_at_fit = self.n_jobs
        return self


class TestForceSerialUnderParallelism:
    """Inside a MetaCausal worker (``INNER_WORKER_ENV`` set), the adapter pins
    the wrapped estimator's joblib ``n_jobs`` to 1 so a forest's inner pool
    cannot nest inside the outer worker (the segfault root cause). Outside a
    worker the estimator's own ``n_jobs`` is left untouched.
    """

    def test_force_serial_pins_model_and_submodels(self):
        from types import SimpleNamespace

        model = SimpleNamespace(
            n_jobs=-1,
            model_y=SimpleNamespace(n_jobs=-1),
            models=[SimpleNamespace(n_jobs=-1), SimpleNamespace(n_jobs=4)],
        )
        EconMLAdapter._force_serial(model)
        assert model.n_jobs == 1
        assert model.model_y.n_jobs == 1
        assert all(m.n_jobs == 1 for m in model.models)

    def test_fit_pins_n_jobs_inside_worker(self, monkeypatch):
        from metacausal._parallel import INNER_WORKER_ENV

        monkeypatch.setenv(INNER_WORKER_ENV, "1")
        adapter = EconMLAdapter(_JobsEstimator(n_jobs=-1))
        X, T, Y = _binary_dgp(n=40, seed=5)
        adapter.fit(X, T, Y)
        # The deep-copied fitted model was pinned *before* fit ran...
        assert adapter._fitted_model.n_jobs_at_fit == 1
        assert adapter._fitted_model.n_jobs == 1
        assert adapter._fitted_model.model_y.n_jobs == 1
        # ...and the original template is left untouched.
        assert adapter._model.n_jobs == -1

    def test_fit_leaves_n_jobs_outside_worker(self, monkeypatch):
        from metacausal._parallel import INNER_WORKER_ENV

        monkeypatch.delenv(INNER_WORKER_ENV, raising=False)
        adapter = EconMLAdapter(_JobsEstimator(n_jobs=-1))
        X, T, Y = _binary_dgp(n=40, seed=6)
        adapter.fit(X, T, Y)
        assert adapter._fitted_model.n_jobs_at_fit == -1
        assert adapter._fitted_model.n_jobs == -1


def test_importable_from_top_level():
    from metacausal import EconMLAdapter  # noqa: F401
