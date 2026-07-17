"""Tests for ``default_methods()``.

The smoke test that actually fits every default component on Lalonde is
marked ``slow`` because the stochtree BCF sampler adds noticeable
wall-clock time. Run with ``pytest -m slow`` to opt in, or
``pytest -m ''`` to run everything.
"""

from __future__ import annotations

import pytest

from metacausal import CausalEnsemble
from metacausal.datasets import load_lalonde
from metacausal.defaults import default_methods, default_outcome_model


def test_default_library_has_nine_components_spanning_four_frameworks():
    """Enumeration check: no fitting, just construction and name verification."""
    methods = default_methods()
    assert len(methods) == 9

    ens = CausalEnsemble(methods=methods)
    names = set(ens.method_names)

    expected = {
        # DoubleML
        "DoubleMLIRM", "DoubleMLPLR",
        # EconML
        "CausalForestDML", "DRLearner",
        "TLearner", "XLearner",
        # CausalML
        "BaseRRegressor", "TMLELearner",
        # stochtree
        "BCF",
    }
    assert names == expected, f"missing={expected - names}, unexpected={names - expected}"


class TestDefaultOutcomeModel:
    def test_continuous_returns_regressor(self):
        model = default_outcome_model("continuous")
        from sklearn.base import is_regressor
        assert is_regressor(model)
        assert not hasattr(model, "predict_proba")

    def test_default_is_continuous(self):
        # No-arg call preserves pre-Phase-1.4 behavior.
        from sklearn.base import is_regressor
        assert is_regressor(default_outcome_model())

    def test_binary_returns_classifier(self):
        model = default_outcome_model("binary")
        from sklearn.base import is_classifier
        assert is_classifier(model)
        assert hasattr(model, "predict_proba")

    def test_invalid_outcome_type_raises(self):
        with pytest.raises(ValueError, match="outcome_type"):
            default_outcome_model("survival")


class TestDefaultMethodsOutcomeType:
    def test_continuous_explicit_matches_default(self):
        names_default = {m.name if hasattr(m, "name") else type(m).__name__
                         for m in default_methods()}
        names_explicit = {m.name if hasattr(m, "name") else type(m).__name__
                          for m in default_methods("continuous")}
        assert names_default == names_explicit

    def test_binary_pool_composition(self):
        methods = default_methods("binary")
        assert len(methods) == 7

        ens = CausalEnsemble(methods=methods)
        names = set(ens.method_names)
        expected = {
            "DoubleMLIRM",
            "CausalForestDML", "DRLearner",
            "BaseTClassifier", "BaseXClassifier",
            "BaseRClassifier",
            "TMLELearner",
        }
        assert names == expected

    def test_binary_components_declare_binary_capability(self):
        ens = CausalEnsemble(methods=default_methods("binary"))
        for m in ens._wrapped_methods:
            assert "binary" in m.supported_outcome_types, (
                f"{m.name} does not support binary outcomes"
            )


class TestRLearnerSerialCrossFit:
    """CausalML's BaseRLearner family defaults ``cv_n_jobs=-1`` (unlike
    EconML's ``n_jobs``, this knob is on by default), which would otherwise
    nest a joblib pool inside MetaCausal's own outer parallelism on every
    fit (issue #47). The shipped default pools pin it to 1 explicitly.
    """

    def test_continuous_pool_r_learner_pinned(self):
        methods = default_methods()
        r_learner = next(m for m in methods if type(m).__name__ == "BaseRRegressor")
        assert r_learner.cv_n_jobs == 1

    def test_binary_pool_r_classifier_pinned(self):
        methods = default_methods("binary")
        r_classifier = next(m for m in methods if type(m).__name__ == "BaseRClassifier")
        assert r_classifier.cv_n_jobs == 1

    def test_invalid_outcome_type_raises(self):
        with pytest.raises(ValueError, match="outcome_type"):
            default_methods("survival")


@pytest.mark.slow
def test_binary_default_ensemble_fits_on_synthetic_dgp():
    """Every binary-pool component fits on a synthetic binary DGP and
    produces a finite ATE within a sensible range of the true RD."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-0.3 * X[:, 0]))
    T = rng.binomial(1, e).astype(int)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] - 1.0 + 0.7 * T)))
    Y = rng.binomial(1, p)

    ens = CausalEnsemble(methods=default_methods("binary"))
    ens.fit(X, T, Y, random_state=0)
    result = ens.ate()

    assert np.isfinite(result.ate)
    # All components produced a finite ATE in a plausible RD range.
    assert len(result.component_estimates) == 7
    for name, est in result.component_estimates.items():
        assert np.isfinite(est.ate), f"{name} produced non-finite ATE"
        assert -1.0 <= est.ate <= 1.0, (
            f"{name} ATE={est.ate} is outside the [-1, 1] risk-difference range"
        )


@pytest.mark.slow
def test_default_ensemble_fits_on_lalonde():
    """Every default component fits on Lalonde and produces a finite ATE."""
    X, T, Y = load_lalonde()

    ens = CausalEnsemble()
    ens.fit(X, T, Y, random_state=42)

    result = ens.ate()

    # Ensemble ATE is finite
    assert result.ate == result.ate  # not NaN
    assert abs(result.ate) < 1e6  # not inf-ish

    # All nine components produced a finite ATE
    assert len(result.component_estimates) == 9
    for name, est in result.component_estimates.items():
        assert est.ate == est.ate, f"{name} returned NaN"
