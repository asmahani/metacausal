"""Tests for public repr()/summary() output on result objects."""

from __future__ import annotations

import numpy as np

from metacausal import CausalEnsemble
from metacausal.aggregation.weights import BootstrapResult, EnsembleWeights
from metacausal.estimators import (
    AteEstimate,
    CateEstimate,
    ComponentAteEstimate,
    ComponentCateEstimate,
)


class _MockEstimator:
    def __init__(self, ate: float, name: str):
        self._ate = ate
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass

    def fit(self, X, T, Y, random_state=None, **kwargs):
        return None

    def ate(self, X=None):
        return ComponentAteEstimate(ate=self._ate)

    def cate(self, X):
        return ComponentCateEstimate(cate=np.full(X.shape[0], self._ate))


def _dummy_data(n: int = 8):
    X = np.arange(float(n)).reshape(-1, 1)
    T = np.array([0, 1] * (n // 2), dtype=float)
    Y = X[:, 0] + T
    return X, T, Y


def test_component_ate_repr_is_compact():
    est = ComponentAteEstimate(ate=0.1234, ci_lower=0.01, ci_upper=0.25)

    rendered = repr(est)

    assert rendered.startswith("ComponentAteEstimate(")
    assert "ci=[" in rendered


def test_component_cate_repr_does_not_dump_array():
    est = ComponentCateEstimate(cate=np.array([0.1, 0.2, 0.3]))

    rendered = repr(est)

    assert rendered.startswith("ComponentCateEstimate(")
    assert "array(" not in rendered
    assert "n=3" in rendered


def test_ate_summary_includes_components_and_spread():
    ate = AteEstimate(
        ate=0.125,
        component_estimates={
            "m0": ComponentAteEstimate(ate=0.1),
            "m1": ComponentAteEstimate(ate=0.15, ci_lower=0.05, ci_upper=0.25),
        },
        aggregation="Median",
    )

    rendered = ate.summary(digits=4, signed=True)

    assert "Ensemble ATE (Median): +0.1250" in rendered
    assert "m0" in rendered
    assert "m1" in rendered
    assert "CI [+0.0500, +0.2500]" in rendered
    assert "Spread (max - min): +0.0500" in rendered


def test_ate_summary_show_ci_false_omits_intervals():
    ate = AteEstimate(
        ate=0.125,
        component_estimates={
            "m0": ComponentAteEstimate(ate=0.1),
            "m1": ComponentAteEstimate(ate=0.15, ci_lower=0.05, ci_upper=0.25),
        },
        aggregation="Median",
    )

    rendered = ate.summary(digits=4, signed=True, show_ci=False)

    assert "CI [" not in rendered                       # native intervals suppressed
    assert "m1" in rendered                             # component row still present
    assert "+0.1500" in rendered                        # its point estimate still shown
    assert "Spread (max - min): +0.0500" in rendered


def test_cate_repr_and_summary_do_not_dump_arrays():
    weights = EnsembleWeights(
        weights=np.array([0.25, 0.75]),
        model_names=["m0", "m1"],
        method="CausalStacking",
    )
    cate = CateEstimate(
        cate=np.array([0.1, 0.4, 0.2, 0.3]),
        component_estimates={
            "m0": ComponentCateEstimate(cate=np.array([0.0, 0.1, 0.2, 0.3])),
            "m1": ComponentCateEstimate(cate=np.array([0.2, 0.3, 0.4, 0.5])),
        },
        aggregation="CausalStacking",
        ensemble_weights=weights,
    )

    rendered = repr(cate)
    summary = cate.summary(digits=4)

    assert rendered.startswith("CateEstimate(")
    assert "array(" not in rendered
    assert "CATE summary (CausalStacking)" in summary
    assert "Component pool (2):" in summary
    assert "Weights:" in summary
    assert "m0" in summary
    assert "m1" in summary


def test_bootstrap_repr_and_summary_render_confidence_level():
    boot = BootstrapResult(
        ate=0.5,
        ate_ci_lower=0.2,
        ate_ci_upper=0.8,
        boot_ates=np.array([0.3, 0.5, 0.7]),
        cate=np.array([0.2, 0.4, 0.6]),
        cate_ci_lower=np.array([0.1, 0.2, 0.3]),
        cate_ci_upper=np.array([0.3, 0.6, 0.9]),
        component_boot_ates={
            "m0": np.array([0.2, 0.4, 0.6]),
            "m1": np.array([0.3, 0.5, 0.7]),
        },
        n_boot=3,
        n_failed=1,
        alpha=0.10,
        aggregation="Median",
        method="subsample",
    )

    rendered = repr(boot)
    summary = boot.summary(digits=4, signed=True)

    assert rendered.startswith("BootstrapResult(")
    assert "level=90%" in rendered
    assert "array(" not in rendered
    assert "Bootstrap ATE summary (subsample, 90% CI, n_boot=3, n_failed=1)" in summary
    assert "Ensemble 90% CI: [+0.2000, +0.8000]" in summary
    assert "CATE: n=3" in summary


def test_ensembleweights_repr_is_compact():
    weights = EnsembleWeights(
        weights=np.array([0.2, 0.3, 0.5]),
        model_names=["a", "b", "c"],
        method="CBA",
    )

    rendered = repr(weights)

    assert rendered.startswith("EnsembleWeights(")
    assert "a=0.200" in rendered
    assert "array(" not in rendered


def test_causalensemble_repr_and_summary_pre_and_post_fit():
    default_ens = CausalEnsemble()
    default_repr = repr(default_ens)
    default_summary = default_ens.summary()

    assert default_repr.startswith("CausalEnsemble(")
    assert "<metacausal.ensemble.CausalEnsemble object at" not in default_repr
    assert "default pool deferred to fit()" in default_repr
    assert "Components: default pool deferred to fit()" in default_summary

    fitted = CausalEnsemble([_MockEstimator(1.0, "m0"), _MockEstimator(2.0, "m1")])
    fitted.fit(*_dummy_data(), random_state=0)
    fitted_repr = repr(fitted)
    fitted_summary = fitted.summary()

    assert "fitted=True" in fitted_repr
    assert "components=[m0, m1]" in fitted_repr
    assert "Components (2):" in fitted_summary
    assert "m0" in fitted_summary
    assert "m1" in fitted_summary


class TestSummaryStrReplDisplay:
    """``summary()`` returns are bare-REPL-friendly: ``repr()`` must match
    ``str()`` (i.e. embedded newlines render rather than showing as literal
    ``\\n``), while still behaving as ordinary ``str`` everywhere else."""

    def test_ate_summary_repr_matches_str(self):
        ate = AteEstimate(
            ate=0.1,
            component_estimates={"m0": ComponentAteEstimate(ate=0.1)},
            aggregation="Median",
        )
        s = ate.summary()
        assert repr(s) == str(s)
        assert isinstance(s, str)

    def test_cate_summary_repr_matches_str(self):
        cate = CateEstimate(
            cate=np.array([0.1, 0.2]),
            component_estimates={"m0": ComponentCateEstimate(cate=np.array([0.1, 0.2]))},
            aggregation="Median",
        )
        s = cate.summary()
        assert repr(s) == str(s)
        assert isinstance(s, str)

    def test_causalensemble_summary_repr_matches_str(self):
        s = CausalEnsemble().summary()
        assert repr(s) == str(s)
        assert isinstance(s, str)

    def test_bootstrap_summary_repr_matches_str(self):
        boot = BootstrapResult(
            ate=0.5,
            ate_ci_lower=0.2,
            ate_ci_upper=0.8,
            boot_ates=np.array([0.3, 0.5, 0.7]),
            cate=None,
            cate_ci_lower=None,
            cate_ci_upper=None,
            component_boot_ates={"m0": np.array([0.2, 0.4, 0.6])},
            n_boot=3,
            n_failed=0,
            alpha=0.10,
            aggregation="Median",
            method="subsample",
        )
        s = boot.summary()
        assert repr(s) == str(s)
        assert isinstance(s, str)
