"""Tests pinning the MetaCausalWarning hierarchy and filter behaviour.

The hierarchy is:

    Warning
    └── MetaCausalWarning
        ├── ComponentWarning
        │   ├── ComponentFailureWarning   (component fit/predict failures)
        │   └── ComponentExclusionWarning (configuration mismatches)
        └── BootstrapWarning              (bootstrap inference reliability)

These tests verify (a) the inheritance is correct, (b) each substantive
emission site emits the right subclass (centralised contract guard so
that future refactors cannot silently revert a category by also
updating the per-feature test), and (c)
``warnings.filterwarnings("error", category=...)`` escalates at the
appropriate granularity without affecting unrelated ``RuntimeWarning``
traffic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from metacausal import (
    BootstrapWarning,
    CausalEnsemble,
    ComponentExclusionWarning,
    ComponentFailureWarning,
    ComponentWarning,
    Median,
    MetaCausalWarning,
)
from metacausal.adapters.generic import GenericATEAdapter, GenericCATEAdapter
from metacausal.aggregation import CausalStacking, CrossFitSplit
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_metacausal_subclasses_warning(self):
        assert issubclass(MetaCausalWarning, Warning)

    def test_component_subclasses_metacausal(self):
        assert issubclass(ComponentWarning, MetaCausalWarning)

    def test_bootstrap_subclasses_metacausal(self):
        assert issubclass(BootstrapWarning, MetaCausalWarning)

    def test_failure_subclasses_component(self):
        assert issubclass(ComponentFailureWarning, ComponentWarning)

    def test_exclusion_subclasses_component(self):
        assert issubclass(ComponentExclusionWarning, ComponentWarning)

    def test_no_metacausal_class_subclasses_runtimewarning(self):
        # Independence from RuntimeWarning is the design contract — users
        # who escalate RuntimeWarning broadly should NOT inadvertently
        # catch metacausal warnings, and vice versa.
        for cls in (
            MetaCausalWarning, ComponentWarning, ComponentFailureWarning,
            ComponentExclusionWarning, BootstrapWarning,
        ):
            assert not issubclass(cls, RuntimeWarning), (
                f"{cls.__name__} must not subclass RuntimeWarning"
            )

    def test_failure_and_exclusion_are_disjoint(self):
        assert not issubclass(ComponentFailureWarning, ComponentExclusionWarning)
        assert not issubclass(ComponentExclusionWarning, ComponentFailureWarning)

    def test_component_and_bootstrap_are_disjoint(self):
        # Bootstrap warnings are not a flavour of component warning — they
        # are about inference reliability, not pool composition.
        assert not issubclass(ComponentWarning, BootstrapWarning)
        assert not issubclass(BootstrapWarning, ComponentWarning)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _good_callable(X, T, Y):
    return 1.0


def _failing_callable(X, T, Y):
    raise RuntimeError("intentional callable failure")


def _dummy_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n)
    Y = rng.normal(size=n)
    return X, T, Y


class _FailingAtFit:
    """Minimal adapter that raises on fit — exercises the fit-time
    component-failure path."""

    name = "bad"
    supports_cate = False
    supported_outcome_types = ("continuous",)

    def fit(self, X, T, Y, **kwargs):
        raise RuntimeError("intentional fit failure")

    def ate(self, X=None):  # pragma: no cover
        raise NotImplementedError

    def cate(self, X):  # pragma: no cover
        raise NotImplementedError

    def validate_outcome_type(self, detected):
        return None


class _AteOnlyMock:
    """Minimal CATE-incapable adapter — exercises the ATE-only
    exclusion path in supervised aggregation."""

    name = "ate_only"
    supports_cate = False
    supported_outcome_types = ("continuous",)

    def fit(self, X, T, Y, **kwargs):
        return None

    def ate(self, X=None):
        return ComponentAteEstimate(ate=0.0)

    def cate(self, X):  # pragma: no cover
        raise NotImplementedError

    def validate_outcome_type(self, detected):
        return None


class _FailingCATEAtFit:
    """CATE-capable adapter that always raises on fit. Used for the
    cross-fitting fold-failure site (which only fires for CATE-capable
    adapters in the supervised path).
    """

    name = "bad_cate"
    supports_cate = True
    supported_outcome_types = ("continuous",)

    def fit(self, X, T, Y, **kwargs):
        raise RuntimeError("intentional fit failure")

    def ate(self, X=None):  # pragma: no cover
        raise NotImplementedError

    def cate(self, X):  # pragma: no cover
        raise NotImplementedError

    def validate_outcome_type(self, detected):
        return None


class _GoodCATE:
    """Trivial CATE-capable adapter. Returns a constant CATE."""

    name = "good_cate"
    supports_cate = True
    supported_outcome_types = ("continuous",)

    def fit(self, X, T, Y, **kwargs):
        self._n_features = X.shape[1] if X.ndim > 1 else 1
        return None

    def ate(self, X=None):
        return ComponentAteEstimate(ate=0.5)

    def cate(self, X):
        return ComponentCateEstimate(cate=np.full(len(X), 0.5))

    def validate_outcome_type(self, detected):
        return None


# ---------------------------------------------------------------------------
# Filter scoping
# ---------------------------------------------------------------------------


class TestFilterScoping:
    """Verify each level of the hierarchy filters at the right granularity."""

    def test_filter_metacausal_catches_failure(self):
        # MetaCausalWarning is the umbrella; escalating at this level
        # should catch any subclass.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MetaCausalWarning)
            ens = CausalEnsemble(methods=[
                GenericATEAdapter(_good_callable, name="good"),
                _FailingAtFit(),
            ])
            with pytest.raises(ComponentFailureWarning):
                ens.fit(*_dummy_data())

    def test_filter_component_catches_failure(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ComponentWarning)
            ens = CausalEnsemble(methods=[
                GenericATEAdapter(_good_callable, name="good"),
                _FailingAtFit(),
            ])
            with pytest.raises(ComponentFailureWarning):
                ens.fit(*_dummy_data())

    def test_filter_failure_catches_failure(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ComponentFailureWarning)
            ens = CausalEnsemble(methods=[
                GenericATEAdapter(_good_callable, name="good"),
                _FailingAtFit(),
            ])
            with pytest.raises(ComponentFailureWarning):
                ens.fit(*_dummy_data())

    def test_filter_exclusion_does_not_catch_failure(self):
        # Failure should NOT escalate when only exclusion is filtered.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ComponentExclusionWarning)
            warnings.simplefilter("default", category=ComponentFailureWarning)
            ens = CausalEnsemble(methods=[
                GenericATEAdapter(_good_callable, name="good"),
                _FailingAtFit(),
            ])
            with pytest.warns(ComponentFailureWarning):
                ens.fit(*_dummy_data())

    def test_filter_bootstrap_does_not_catch_component(self):
        # BootstrapWarning is a sibling of ComponentWarning, not a parent
        # — escalating BootstrapWarning should NOT escalate component
        # failures.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=BootstrapWarning)
            warnings.simplefilter("default", category=ComponentFailureWarning)
            ens = CausalEnsemble(methods=[
                GenericATEAdapter(_good_callable, name="good"),
                _FailingAtFit(),
            ])
            with pytest.warns(ComponentFailureWarning):
                ens.fit(*_dummy_data())

    def test_filter_component_does_not_catch_bootstrap(self, monkeypatch):
        # ComponentWarning is a sibling of BootstrapWarning — escalating
        # ComponentWarning should NOT escalate BootstrapWarning.
        ens = CausalEnsemble(methods=[
            GenericATEAdapter(_good_callable, name="good"),
        ])
        ens.fit(*_dummy_data(), random_state=0)

        # Force every replicate to fail so the all-replicates-failed
        # warning fires.
        monkeypatch.setattr(
            CausalEnsemble, "_single_bootstrap",
            lambda self, X_eval, seed: None,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ComponentWarning)
            warnings.simplefilter("default", category=BootstrapWarning)
            with pytest.warns(BootstrapWarning):
                ens.bootstrap(n_boot=5, random_state=0)

    def test_filter_runtimewarning_does_not_catch_metacausal(self):
        # The whole point of inheriting from Warning (not RuntimeWarning)
        # is that RuntimeWarning policies do not affect metacausal warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.simplefilter("default", category=ComponentFailureWarning)
            ens = CausalEnsemble(methods=[
                GenericATEAdapter(_good_callable, name="good"),
                _FailingAtFit(),
            ])
            with pytest.warns(ComponentFailureWarning):
                ens.fit(*_dummy_data())


# ---------------------------------------------------------------------------
# Site-level emission contract
# ---------------------------------------------------------------------------


class TestEmissionSites:
    """Centralised contract: each substantive site emits the correct
    warning class. A regression that flips a site's category back to
    RuntimeWarning will fail at least one of these tests, even if the
    feature-specific test in another file has been updated to match.
    """

    def test_fit_failure_emits_failure_warning(self):
        # Site: ensemble.py:432 — adapter raises on full-data fit.
        ens = CausalEnsemble(methods=[
            GenericATEAdapter(_good_callable, name="good"),
            _FailingAtFit(),
        ])
        with pytest.warns(ComponentFailureWarning, match=r"Method 'bad' failed"):
            ens.fit(*_dummy_data())

    def test_outcome_type_filter_emits_exclusion_warning(self):
        # Site: ensemble.py:278 — outcome-type filter drops a component.
        cont_only = GenericATEAdapter(
            _good_callable, name="cont_only", supported_outcome_types=("continuous",),
        )
        both = GenericATEAdapter(
            _good_callable, name="both", supported_outcome_types=("continuous", "binary"),
        )
        ens = CausalEnsemble(methods=[cont_only, both])
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        T = rng.binomial(1, 0.5, size=50)
        Y = rng.binomial(1, 0.5, size=50)
        with pytest.warns(ComponentExclusionWarning, match="Outcome type"):
            ens.fit(X, T, Y)

    def test_ate_only_in_supervised_emits_exclusion_warning(self):
        # Site: ensemble.py:477 — ATE-only adapter dropped from supervised path.
        ens = CausalEnsemble(
            methods=[_GoodCATE(), _AteOnlyMock()],
            aggregation=CausalStacking(split=CrossFitSplit(n_folds=2)),
        )
        with pytest.warns(
            ComponentExclusionWarning, match="ATE-only adapters will be skipped"
        ):
            ens.fit(*_dummy_data(n=80, seed=0), random_state=0)

    def test_cross_fit_fold_failure_emits_failure_warning(self):
        # Site: ensemble.py:588 — adapter fails during cross-fitting fold.
        ens = CausalEnsemble(
            methods=[_GoodCATE(), _FailingCATEAtFit()],
            aggregation=CausalStacking(split=CrossFitSplit(n_folds=2)),
        )
        with pytest.warns(
            ComponentFailureWarning, match="Dropping from all folds"
        ):
            ens.fit(*_dummy_data(n=80, seed=0), random_state=0)

    def test_predict_failure_emits_failure_warning(self):
        # Site: ensemble.py:807 — adapter raises during ate(). Use
        # GenericATEAdapter with a failing callable: the callable runs
        # at ate()-time, not fit()-time, so the failure surfaces during
        # the predict step.
        ens = CausalEnsemble(methods=[
            GenericATEAdapter(_good_callable, name="good"),
            GenericATEAdapter(_failing_callable, name="bad_predict"),
        ])
        ens.fit(*_dummy_data(), random_state=0)
        with pytest.warns(
            ComponentFailureWarning, match=r"Method 'bad_predict' failed during ate\(\)"
        ):
            ens.ate()

    def test_bootstrap_all_replicates_failed_emits_bootstrap_warning(
        self, monkeypatch
    ):
        # Site: ensemble.py:1001 — all bootstrap replicates failed.
        # Patch _single_bootstrap to force every replicate to fail.
        ens = CausalEnsemble(methods=[
            GenericATEAdapter(_good_callable, name="good"),
        ])
        ens.fit(*_dummy_data(), random_state=0)
        monkeypatch.setattr(
            CausalEnsemble, "_single_bootstrap",
            lambda self, X_eval, seed: None,
        )
        with pytest.warns(BootstrapWarning, match="All bootstrap replicates failed"):
            ens.bootstrap(n_boot=5, random_state=0)

    def test_bootstrap_partial_failures_emits_bootstrap_warning(
        self, monkeypatch
    ):
        # Site: ensemble.py:993 — >10% of replicates failed.
        # Patch _single_bootstrap so half the replicates fail (well
        # above the 10% threshold) but enough succeed to populate
        # `valid` and trigger the threshold-warning branch rather than
        # the all-failed branch.
        ens = CausalEnsemble(methods=[
            GenericATEAdapter(_good_callable, name="good"),
        ])
        ens.fit(*_dummy_data(), random_state=0)

        # Alternate failure / success.
        call_counter = {"n": 0}
        def alternating(self, X_eval, seed):
            call_counter["n"] += 1
            if call_counter["n"] % 2 == 0:
                return None
            # Return a valid (ate, comp_ates, cate) triple.
            return 1.0, {"good": 1.0}, None

        monkeypatch.setattr(CausalEnsemble, "_single_bootstrap", alternating)
        with pytest.warns(BootstrapWarning, match="bootstrap replicates failed"):
            ens.bootstrap(n_boot=10, random_state=0)
