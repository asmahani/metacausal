"""Tier-2 integration tests: real upstream libraries on small synthetic data.

Runs by default alongside the rest of the suite. Select with
``pytest -m integration`` to run only this tier (e.g. as a localized
signal in Dependabot CI when an upstream version bumps).

This file has two purposes:

1. Per-adapter round-trips for libraries where dedicated test files
   either don't exist (EconML) or are entirely mocked
   (``test_stochtree_adapter.py``). DoubleML round-trips extend the
   reproducibility-focused ``test_doubleml_adapter.py``.

2. Targeted upstream-API *contract* assertions — pin specific upstream
   behaviors our adapters depend on, so that a Dependabot bump that
   shifts the upstream surface fails here with a localized message
   ("``CausalForestDML.ate(X)`` shape changed from ``(1,)``") instead of
   producing cryptic adapter errors elsewhere. Each contract test names
   the adapter file that depends on the pinned behavior.

CausalML adapter tests already exercise real CausalML thoroughly in
``test_causalml_adapter.py``; only a couple of contract assertions are
duplicated here for the same Dependabot-signal reason.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression

from metacausal import CausalEnsemble

# Every test in this file touches at least one real upstream library.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _continuous_dgp(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(int)
    Y = X[:, 0] + 0.5 * T + rng.normal(scale=0.5, size=n)
    return X, T, Y


def _binary_dgp(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n).astype(int)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] + 0.7 * T)))
    Y = rng.binomial(1, p)
    return X, T, Y


# ---------------------------------------------------------------------------
# DoubleML — round-trip on real upstream
# ---------------------------------------------------------------------------


class TestDoubleMLRoundTrip:
    def test_irm_continuous(self):
        from doubleml import DoubleMLIRM

        from metacausal.adapters.doubleml import DoubleMLAdapter

        adapter = DoubleMLAdapter(
            DoubleMLIRM,
            ml_g=HistGradientBoostingRegressor(max_iter=50),
            ml_m=LogisticRegression(max_iter=500),
        )
        X, T, Y = _continuous_dgp()
        adapter.fit(X, T, Y, random_state=0)
        result = adapter.ate()
        assert isinstance(result.ate, float)
        assert np.isfinite(result.ate)

    def test_irm_binary(self):
        from doubleml import DoubleMLIRM

        from metacausal.adapters.doubleml import DoubleMLAdapter

        adapter = DoubleMLAdapter(
            DoubleMLIRM,
            ml_g=HistGradientBoostingClassifier(max_iter=50),
            ml_m=LogisticRegression(max_iter=500),
        )
        X, T, Y = _binary_dgp()
        adapter.fit(X, T, Y, random_state=0)
        result = adapter.ate()
        assert isinstance(result.ate, float)
        assert np.isfinite(result.ate)
        assert -1.0 <= result.ate <= 1.0  # risk difference

    def test_plr_continuous(self):
        from doubleml import DoubleMLPLR

        from metacausal.adapters.doubleml import DoubleMLAdapter

        adapter = DoubleMLAdapter(
            DoubleMLPLR,
            ml_l=HistGradientBoostingRegressor(max_iter=50),
            ml_m=LogisticRegression(max_iter=500),
        )
        X, T, Y = _continuous_dgp()
        adapter.fit(X, T, Y, random_state=0)
        result = adapter.ate()
        assert isinstance(result.ate, float)
        assert np.isfinite(result.ate)


# ---------------------------------------------------------------------------
# EconML — per-class round-trips (no dedicated adapter test file today)
# ---------------------------------------------------------------------------


class TestEconMLRoundTripContinuous:
    def test_causal_forest_dml(self):
        from econml.dml import CausalForestDML

        model = CausalForestDML(
            model_y=HistGradientBoostingRegressor(max_iter=50),
            model_t=HistGradientBoostingClassifier(max_iter=50),
            discrete_treatment=True,
            n_estimators=20,
        )
        ens = CausalEnsemble(methods=[model])
        X, T, Y = _continuous_dgp()
        ens.fit(X, T, Y, random_state=0)
        result = ens.ate()
        assert np.isfinite(result.ate)
        cate = ens.cate(X)
        assert cate.cate.shape == (X.shape[0],)
        assert np.issubdtype(cate.cate.dtype, np.floating)

    def test_dr_learner(self):
        from econml.dr import DRLearner

        model = DRLearner(
            model_regression=HistGradientBoostingRegressor(max_iter=50),
            model_propensity=LogisticRegression(max_iter=500),
        )
        ens = CausalEnsemble(methods=[model])
        X, T, Y = _continuous_dgp()
        ens.fit(X, T, Y, random_state=0)
        assert np.isfinite(ens.ate().ate)

    def test_slearner(self):
        from econml.metalearners import SLearner

        model = SLearner(overall_model=HistGradientBoostingRegressor(max_iter=50))
        ens = CausalEnsemble(methods=[model])
        X, T, Y = _continuous_dgp()
        ens.fit(X, T, Y, random_state=0)
        assert np.isfinite(ens.ate().ate)

    def test_tlearner(self):
        from econml.metalearners import TLearner

        model = TLearner(models=HistGradientBoostingRegressor(max_iter=50))
        ens = CausalEnsemble(methods=[model])
        X, T, Y = _continuous_dgp()
        ens.fit(X, T, Y, random_state=0)
        assert np.isfinite(ens.ate().ate)

    def test_xlearner(self):
        from econml.metalearners import XLearner

        model = XLearner(
            models=HistGradientBoostingRegressor(max_iter=50),
            propensity_model=LogisticRegression(max_iter=500),
        )
        ens = CausalEnsemble(methods=[model])
        X, T, Y = _continuous_dgp()
        ens.fit(X, T, Y, random_state=0)
        assert np.isfinite(ens.ate().ate)


class TestEconMLRoundTripBinary:
    def test_causal_forest_dml(self):
        from econml.dml import CausalForestDML

        model = CausalForestDML(
            model_y=HistGradientBoostingClassifier(max_iter=50),
            model_t=HistGradientBoostingClassifier(max_iter=50),
            discrete_outcome=True,
            discrete_treatment=True,
            n_estimators=20,
        )
        ens = CausalEnsemble(methods=[model])
        X, T, Y = _binary_dgp()
        ens.fit(X, T, Y, random_state=0)
        result = ens.ate()
        assert np.isfinite(result.ate)
        assert -1.0 <= result.ate <= 1.0

    def test_dr_learner(self):
        from econml.dr import DRLearner

        model = DRLearner(
            model_regression=HistGradientBoostingClassifier(max_iter=50),
            model_propensity=LogisticRegression(max_iter=500),
            discrete_outcome=True,
        )
        ens = CausalEnsemble(methods=[model])
        X, T, Y = _binary_dgp()
        ens.fit(X, T, Y, random_state=0)
        assert np.isfinite(ens.ate().ate)


# ---------------------------------------------------------------------------
# Stochtree — real BCF (test_stochtree_adapter.py is fully mocked)
# ---------------------------------------------------------------------------


class TestStochtreeRoundTrip:
    def test_bcf_continuous(self):
        from metacausal.adapters.stochtree import StochtreeAdapter

        adapter = StochtreeAdapter(
            propensity_model=LogisticRegression(max_iter=500),
            num_gfr=2,
            num_burnin=20,
            num_mcmc=20,
        )
        X, T, Y = _continuous_dgp()
        adapter.fit(X, T, Y, random_state=0)
        result = adapter.ate()
        assert np.isfinite(result.ate)
        cate = adapter.cate(X)
        assert cate.cate.shape == (X.shape[0],)
        assert np.issubdtype(cate.cate.dtype, np.floating)


# ---------------------------------------------------------------------------
# Upstream API contracts — pin behaviors our adapters depend on. A failure
# here points at upstream API drift; the docstring names the affected
# adapter file so a fix is localised.
# ---------------------------------------------------------------------------


class TestEconMLContracts:
    """EconMLAdapter (metacausal/adapters/econml.py)."""

    def test_continuous_ate_returns_0d(self):
        """Continuous outcome: ``model.ate(X)`` is a 0-d ``np.float64``.
        ``EconMLAdapter._scalar`` flattens for the binary (1-d) case;
        the continuous case relies on this 0-d output to round-trip
        cleanly."""
        from econml.dml import CausalForestDML

        model = CausalForestDML(
            model_y=HistGradientBoostingRegressor(max_iter=50),
            model_t=HistGradientBoostingClassifier(max_iter=50),
            discrete_treatment=True,
            n_estimators=20,
        )
        X, T, Y = _continuous_dgp()
        model.fit(Y, T, X=X)
        ate = model.ate(X=X)
        assert np.asarray(ate).ndim == 0, (
            f"CausalForestDML.ate(X) is no longer 0-d for continuous Y; "
            f"shape is {np.asarray(ate).shape}. Review "
            f"metacausal/adapters/econml.py::EconMLAdapter.ate."
        )

    def test_discrete_outcome_ate_returns_1d(self):
        """Binary outcome: ``model.ate(X)`` is a ``(1,)`` array. NumPy 2.x
        rejects ``float()`` on 1-d arrays, so ``EconMLAdapter._scalar``
        flattens before scalar conversion. If this test starts seeing 0-d
        for binary, the workaround is no longer needed."""
        from econml.dml import CausalForestDML

        model = CausalForestDML(
            model_y=HistGradientBoostingClassifier(max_iter=50),
            model_t=HistGradientBoostingClassifier(max_iter=50),
            discrete_outcome=True,
            discrete_treatment=True,
            n_estimators=20,
        )
        X, T, Y = _binary_dgp()
        model.fit(Y, T, X=X)
        ate = model.ate(X=X)
        assert np.asarray(ate).shape == (1,), (
            f"CausalForestDML(discrete_outcome=True).ate(X) shape changed "
            f"from (1,) to {np.asarray(ate).shape}. Review "
            f"metacausal/adapters/econml.py::EconMLAdapter.ate::_scalar."
        )

    def test_drlearner_has_discrete_outcome_param(self):
        """``DRLearner`` accepts ``discrete_outcome``. ``EconMLAdapter._BINARY_CAPABLE_SLOTS``
        depends on this parameter being present in the constructor."""
        from econml.dr import DRLearner

        sig = inspect.signature(DRLearner.__init__)
        assert "discrete_outcome" in sig.parameters

    def test_causal_forest_dml_has_discrete_outcome_param(self):
        """``CausalForestDML`` accepts ``discrete_outcome``."""
        from econml.dml import CausalForestDML

        sig = inspect.signature(CausalForestDML.__init__)
        assert "discrete_outcome" in sig.parameters


class TestDoubleMLContracts:
    """DoubleMLAdapter (metacausal/adapters/doubleml.py)."""

    def test_data_auto_detects_binary_outcome(self):
        """``DoubleMLData`` auto-sets ``binary_outcome=True`` when Y values
        are in ``{0, 1}``. ``DoubleMLAdapter.fit`` never sets the flag
        explicitly; if this auto-detection breaks, the adapter will fit
        binary IRM as a regression and produce wrong estimates."""
        from doubleml import DoubleMLData

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "X0": rng.normal(size=20),
            "T": rng.binomial(1, 0.5, size=20),
            "Y": rng.binomial(1, 0.5, size=20),
        })
        data = DoubleMLData(df, y_col="Y", d_cols="T", x_cols=["X0"])
        # ``binary_outcome`` may be a numpy bool; check truthiness, not identity.
        assert bool(data.binary_outcome) is True

    def test_irm_routes_classifier_ml_g_to_predict_proba(self):
        """``DoubleMLIRM`` accepts a classifier as ``ml_g`` when Y is binary
        and routes prediction through ``predict_proba``. The adapter does
        not inspect this routing — it passes ``ml_g`` through unchanged
        and trusts IRM to do the right thing."""
        from doubleml import DoubleMLData, DoubleMLIRM

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "X0": rng.normal(size=200),
            "T": rng.binomial(1, 0.5, size=200),
            "Y": rng.binomial(1, 0.5, size=200),
        })
        data = DoubleMLData(df, y_col="Y", d_cols="T", x_cols=["X0"])
        model = DoubleMLIRM(
            data,
            ml_g=HistGradientBoostingClassifier(max_iter=50),
            ml_m=LogisticRegression(max_iter=500),
        )
        model.fit()  # fit succeeds → routing is in place
        assert hasattr(model, "coef")


class TestCausalMLContracts:
    """CausalMLAdapter (metacausal/adapters/causalml.py)."""

    def test_module_namespace(self):
        """Meta-learner classes live under the ``causalml.inference.*``
        prefix. ``CausalMLAdapter._is_causalml`` matches on this prefix to
        decide whether to wrap an instance."""
        from causalml.inference.meta import BaseSRegressor

        instance = BaseSRegressor(learner=HistGradientBoostingRegressor(max_iter=10))
        assert type(instance).__module__.startswith("causalml.inference")

    def test_classifier_predict_proba_shape(self):
        """sklearn classifiers return a ``(n, 2)`` matrix from
        ``predict_proba`` on binary targets. CausalML's ``Base*Classifier``
        pipelines (and our ``fit_nuisance`` binary path) read column ``[:, 1]``
        — a shape change here would silently misread risk-difference
        estimates."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = rng.binomial(1, 0.5, size=50)
        clf = HistGradientBoostingClassifier(max_iter=10).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.ndim == 2 and proba.shape[1] == 2


class TestStochtreeContracts:
    """StochtreeAdapter (metacausal/adapters/stochtree.py)."""

    def test_bcf_sample_signature(self):
        """``BCFModel.sample`` accepts ``X_train``, ``Z_train``, ``y_train``,
        and ``propensity_train``. ``StochtreeAdapter.fit`` calls these by
        keyword; a renamed argument in stochtree would break the adapter."""
        from stochtree import BCFModel

        sig = inspect.signature(BCFModel.sample)
        for kw in ("X_train", "Z_train", "y_train", "propensity_train"):
            assert kw in sig.parameters, (
                f"BCFModel.sample no longer accepts {kw!r}. Review "
                f"metacausal/adapters/stochtree.py::StochtreeAdapter.fit."
            )
