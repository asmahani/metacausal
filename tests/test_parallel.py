"""Tests for n_jobs parallelism in CausalEnsemble (issue #10).

These verify:
- Seed equivalence between sequential and parallel execution.
- Drop-entirely failure semantics in cross-fitting.
- End-to-end smoke test with a real threaded estimator (sklearn HGB)
  to catch oversubscription / deadlock regressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("joblib")

from metacausal import CausalEnsemble, ComponentFailureWarning, Median
from metacausal.aggregation import CrossFitSplit, SupervisedStrategy
from metacausal.aggregation.weights import EnsembleWeights
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class SeededAdapter:
    """ATE-only adapter whose output depends deterministically on random_state."""

    def __init__(self, name: str, offset: float = 0.0):
        self._name = name
        self._offset = offset
        self._ate: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, random_state=None, **kwargs):
        rng = np.random.default_rng(random_state)
        self._ate = self._offset + float(rng.normal())

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._ate)

    def cate(self, X):
        raise NotImplementedError

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class SeededCateAdapter:
    """CATE-capable adapter whose output depends deterministically on random_state."""

    def __init__(self, name: str, offset: float = 0.0):
        self._name = name
        self._offset = offset
        self._shift: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, random_state=None, **kwargs):
        rng = np.random.default_rng(random_state)
        self._shift = self._offset + float(rng.normal())

    def ate(self, X=None):
        n = len(X) if X is not None else 1
        return ComponentAteEstimate(ate=self._shift)

    def cate(self, X):
        return ComponentCateEstimate(cate=self._shift * np.ones(X.shape[0]))

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class AlwaysFailingCateAdapter:
    """CATE-capable adapter whose fit() always raises."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, random_state=None, **kwargs):
        raise ValueError("Intentional failure")

    def ate(self, X=None):
        return ComponentAteEstimate(ate=0.0)

    def cate(self, X):
        return ComponentCateEstimate(cate=np.zeros(X.shape[0]))

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


@dataclass
class UniformSupervisedStrategy(SupervisedStrategy):
    """Supervised strategy with uniform weights — for testing only."""

    def fit_weights(self, cate_predictions, Y, T, X, nuisance) -> EnsembleWeights:
        K = cate_predictions.shape[0]
        weights = np.ones(K) / K
        self._weights = EnsembleWeights(
            weights=weights,
            model_names=[],
            method="uniform_mock",
        )
        return self._weights


def _supervised_strategy(n_folds: int = 3) -> UniformSupervisedStrategy:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    return UniformSupervisedStrategy(
        split=CrossFitSplit(n_folds=n_folds, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
    )


def _dgp(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    Y = X[:, 0] + T * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, T, Y


# ---------------------------------------------------------------------------
# Seed equivalence: simple fit path
# ---------------------------------------------------------------------------


class TestSimpleFitSeedEquivalence:
    """Non-supervised fit: n_jobs=2 must produce identical ATE to n_jobs=1."""

    def _methods(self):
        return [SeededAdapter(f"m{i}", offset=float(i)) for i in range(4)]

    def test_ate_identical_across_n_jobs(self):
        X, T, Y = _dgp()

        ens1 = CausalEnsemble(methods=self._methods(), aggregation=Median())
        ens1.fit(X, T, Y, random_state=42, n_jobs=1)

        ens2 = CausalEnsemble(methods=self._methods(), aggregation=Median())
        ens2.fit(X, T, Y, random_state=42, n_jobs=2)

        ate1 = ens1.ate()
        ate2 = ens2.ate()
        assert ate1.ate == pytest.approx(ate2.ate)
        for name in ate1.component_estimates:
            assert ate1.component_estimates[name].ate == pytest.approx(
                ate2.component_estimates[name].ate
            )


# ---------------------------------------------------------------------------
# Seed equivalence: supervised fit path
# ---------------------------------------------------------------------------


class TestSupervisedFitSeedEquivalence:
    """Supervised fit: n_jobs=2 must match n_jobs=1 including cached OOF predictions."""

    def _methods(self):
        return [
            SeededCateAdapter("a", offset=0.0),
            SeededCateAdapter("b", offset=1.0),
            SeededCateAdapter("c", offset=2.0),
        ]

    def test_oof_predictions_identical(self):
        X, T, Y = _dgp(n=300)

        ens1 = CausalEnsemble(
            methods=self._methods(), aggregation=_supervised_strategy()
        )
        ens1.fit(X, T, Y, random_state=7, n_jobs=1)

        ens2 = CausalEnsemble(
            methods=self._methods(), aggregation=_supervised_strategy()
        )
        ens2.fit(X, T, Y, random_state=7, n_jobs=2)

        assert ens1._cached_oof_cate_model_names == ens2._cached_oof_cate_model_names
        np.testing.assert_allclose(
            ens1._cached_oof_cate_predictions,
            ens2._cached_oof_cate_predictions,
        )

    def test_ate_identical(self):
        X, T, Y = _dgp(n=300)

        ens1 = CausalEnsemble(
            methods=self._methods(), aggregation=_supervised_strategy()
        )
        ens1.fit(X, T, Y, random_state=7, n_jobs=1)

        ens2 = CausalEnsemble(
            methods=self._methods(), aggregation=_supervised_strategy()
        )
        ens2.fit(X, T, Y, random_state=7, n_jobs=2)

        assert ens1.ate().ate == pytest.approx(ens2.ate().ate)


# ---------------------------------------------------------------------------
# Seed equivalence: bootstrap
# ---------------------------------------------------------------------------


class TestBootstrapSeedEquivalence:
    def test_boot_ates_identical(self):
        X, T, Y = _dgp()
        methods = [SeededAdapter(f"m{i}", offset=float(i)) for i in range(3)]

        ens1 = CausalEnsemble(methods=methods, aggregation=Median())
        r1 = ens1.estimate(X, T, Y, n_boot=8, random_state=123, n_jobs=1)

        ens2 = CausalEnsemble(methods=methods, aggregation=Median())
        r2 = ens2.estimate(X, T, Y, n_boot=8, random_state=123, n_jobs=2)

        np.testing.assert_allclose(r1.boot_ates, r2.boot_ates)
        assert r1.ate == pytest.approx(r2.ate)


# ---------------------------------------------------------------------------
# Drop-entirely semantics in parallel cross-fitting
# ---------------------------------------------------------------------------


class TestCrossFittingDropEntirely:
    def test_failing_method_absent_from_oof(self):
        """A method that fails during cross-fitting is dropped entirely, and
        surviving methods are unaffected."""
        X, T, Y = _dgp()
        methods = [
            SeededCateAdapter("ok1", offset=0.0),
            AlwaysFailingCateAdapter("bad"),
            SeededCateAdapter("ok2", offset=1.0),
        ]
        ens = CausalEnsemble(methods=methods, aggregation=_supervised_strategy())

        with pytest.warns(ComponentFailureWarning, match="Method 'bad' failed"):
            ens.fit(X, T, Y, random_state=0, n_jobs=2)

        assert "bad" not in ens._cached_oof_cate_model_names
        assert "ok1" in ens._cached_oof_cate_model_names
        assert "ok2" in ens._cached_oof_cate_model_names

    def test_single_warning_per_failing_method(self):
        """In parallel, each failing method emits exactly one dropped warning
        — even though multiple folds may have failed."""
        import warnings

        X, T, Y = _dgp()
        methods = [
            SeededCateAdapter("ok", offset=0.0),
            AlwaysFailingCateAdapter("bad"),
        ]
        ens = CausalEnsemble(methods=methods, aggregation=_supervised_strategy())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ens.fit(X, T, Y, random_state=0, n_jobs=2)

        dropped_warnings = [
            w for w in caught
            if "Method 'bad' failed" in str(w.message)
            and "Dropping from all folds" in str(w.message)
        ]
        assert len(dropped_warnings) == 1


# ---------------------------------------------------------------------------
# Smoke: real threaded estimator under parallel outer
# ---------------------------------------------------------------------------


def _hgb_fit(X, T, Y, **kwargs):
    """Module-level so the adapter is pickle-able by loky workers."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    m1 = HistGradientBoostingRegressor(max_iter=30).fit(X[T == 1], Y[T == 1])
    m0 = HistGradientBoostingRegressor(max_iter=30).fit(X[T == 0], Y[T == 0])
    return (m0, m1)


def _hgb_cate(state, X):
    m0, m1 = state
    return m1.predict(X) - m0.predict(X)


def test_pin_threads_sets_inner_worker_sentinel():
    """_pin_threads (the worker initializer) must set INNER_WORKER_ENV so
    adapters can pin joblib-based inner parallelism, alongside the BLAS/OMP
    thread vars. Restores the process environment afterwards."""
    import os

    from metacausal._parallel import (
        INNER_WORKER_ENV,
        _THREAD_PIN_VARS,
        _pin_threads,
    )

    watched = (*_THREAD_PIN_VARS, INNER_WORKER_ENV)
    saved = {k: os.environ.get(k) for k in watched}
    try:
        os.environ.pop(INNER_WORKER_ENV, None)
        _pin_threads()
        assert os.environ.get(INNER_WORKER_ENV) == "1"
        for var in _THREAD_PIN_VARS:
            assert os.environ.get(var) == "1"
    finally:
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


def test_inner_worker_env_is_publicly_exported():
    """INNER_WORKER_ENV must be importable from the public metacausal.adapters
    namespace (issue #49) -- not just metacausal._parallel -- so that authors
    of custom components (GenericCATEAdapter, arbitrary callables) have a
    supported way to cooperate with MetaCausal's own-worker parallelism
    guard without reaching into a private module."""
    from metacausal._parallel import INNER_WORKER_ENV as _private
    from metacausal.adapters import INNER_WORKER_ENV as _public

    assert _public == _private == "METACAUSAL_INNER_WORKER"
    assert "INNER_WORKER_ENV" in __import__("metacausal.adapters", fromlist=["__all__"]).__all__


class TestSklearnHGBSmoke:
    """Exercise the thread-pinning / oversubscription fix with sklearn HGB."""

    def test_supervised_fit_parallel_completes(self):
        """HistGradientBoosting + parallel cross-fitting completes without
        hanging. Deadlocks would manifest as a pytest timeout rather than a
        failed assertion — we deliberately keep the workload small."""
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        from metacausal.adapters import GenericCATEAdapter

        X, T, Y = _dgp(n=400)
        methods = [
            GenericCATEAdapter(_hgb_fit, _hgb_cate, name=f"hgb_{i}")
            for i in range(2)
        ]
        ens = CausalEnsemble(
            methods=methods,
            aggregation=UniformSupervisedStrategy(
                split=CrossFitSplit(n_folds=3, stratify=False),
                propensity_model=HistGradientBoostingClassifier(max_iter=30),
                outcome_model=HistGradientBoostingRegressor(max_iter=30),
            ),
        )
        ens.fit(X, T, Y, random_state=0, n_jobs=2)
        assert ens._is_fitted
        _ = ens.ate()
