"""Tests for metacausal.plots and BootstrapResult.component_ate_summary."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from metacausal import CausalEnsemble
from metacausal.aggregation import CrossFitSplit
from metacausal.aggregation.weights import BootstrapResult, EnsembleWeights
from metacausal.estimators import (
    CateEstimate,
    ComponentAteEstimate,
    ComponentCateEstimate,
)
from metacausal.plots import cate_profile, disagreement, forest, weights
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.test_supervised_bootstrap import _fitted_supervised_ensemble
from tests.test_supervised_fit import UniformSupervisedStrategy, _dgp


class _HeterogeneousCateAdapter:
    """Mock CATE-capable adapter with unit-specific, seed-controlled CATE.

    Unlike the shared ``CateCapableAdapter`` (which returns constant
    CATE), this one gives each unit a distinct treatment effect so that
    pairwise correlations are well-defined.
    """

    def __init__(self, name: str, seed: int):
        self._name = name
        self._seed = seed
        self._fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, random_state=None, **kwargs):
        self._fitted = True

    def ate(self, X=None):
        return ComponentAteEstimate(ate=0.0)

    def cate(self, X):
        rng = np.random.default_rng(self._seed)
        return ComponentCateEstimate(
            cate=X[:, 0] + rng.normal(0, 0.3, size=X.shape[0])
        )

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


def _fitted_hetero_ensemble(n: int = 120, n_methods: int = 3, seed: int = 0):
    X, T, Y = _dgp(n=n, seed=seed)
    methods = [
        _HeterogeneousCateAdapter(f"m{i}", seed=seed + i)
        for i in range(n_methods)
    ]
    strategy = UniformSupervisedStrategy(
        split=CrossFitSplit(n_folds=3, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
    )
    ens = CausalEnsemble(methods=methods, aggregation=strategy)
    ens.fit(X, T, Y, random_state=seed)
    return ens, X


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def boot_result_with_components():
    """A BootstrapResult with per-component ATE distributions.

    Built directly (not via ensemble.bootstrap()) so the test is fast
    and deterministic.
    """
    rng = np.random.default_rng(0)
    return BootstrapResult(
        ate=0.5,
        ate_ci_lower=0.1,
        ate_ci_upper=0.9,
        boot_ates=rng.normal(0.5, 0.2, size=50),
        cate=None,
        cate_ci_lower=None,
        cate_ci_upper=None,
        component_boot_ates={
            "m0": rng.normal(0.3, 0.3, size=50),
            "m1": rng.normal(0.7, 0.25, size=50),
            "m2": rng.normal(0.5, 0.5, size=50),
        },
        n_boot=50,
        alpha=0.05,
        aggregation="Mock",
    )


@pytest.fixture
def cate_grid_ensemble():
    """Lightweight fitted ensemble with non-constant CATE per component."""
    return _fitted_hetero_ensemble(n=120, n_methods=3, seed=0)


# ---------------------------------------------------------------------------
# BootstrapResult.component_ate_summary
# ---------------------------------------------------------------------------


class TestComponentAteSummary:
    def test_returns_dataframe_with_expected_columns(
        self, boot_result_with_components
    ):
        df = boot_result_with_components.component_ate_summary()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "mean", "lo", "hi"]
        assert len(df) == 3

    def test_preserves_insertion_order(self, boot_result_with_components):
        df = boot_result_with_components.component_ate_summary()
        assert df["name"].tolist() == ["m0", "m1", "m2"]

    def test_ci_bounds_ordered(self, boot_result_with_components):
        df = boot_result_with_components.component_ate_summary()
        assert (df["lo"] <= df["mean"]).all()
        assert (df["mean"] <= df["hi"]).all()

    def test_raises_on_empty(self):
        boot = BootstrapResult(
            ate=0.0, ate_ci_lower=0.0, ate_ci_upper=0.0,
            boot_ates=np.zeros(1),
            cate=None, cate_ci_lower=None, cate_ci_upper=None,
        )
        with pytest.raises(ValueError, match="component_boot_ates is empty"):
            boot.component_ate_summary()


# ---------------------------------------------------------------------------
# forest()
# ---------------------------------------------------------------------------


class TestForest:
    def test_returns_axes(self, boot_result_with_components):
        ax = forest(boot_result_with_components)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_draws_one_row_per_component_plus_ensemble(
        self, boot_result_with_components
    ):
        ax = forest(boot_result_with_components)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert len(labels) == 4  # 3 components + ensemble
        assert labels[-1] == "Ensemble"
        plt.close(ax.figure)

    @pytest.mark.parametrize("order", ["value", "alpha", "ci_width", "input"])
    def test_ordering_runs(self, boot_result_with_components, order):
        ax = forest(boot_result_with_components, order=order)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_order_value_sorts_by_mean(self, boot_result_with_components):
        ax = forest(boot_result_with_components, order="value")
        # Y-ticks go bottom-to-top; labels list is already in that order.
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # Ensemble at top (last). Components below should be sorted
        # ascending by bootstrap-mean ATE.
        summary = boot_result_with_components.component_ate_summary()
        expected_order = (
            summary.sort_values("mean")["name"].tolist() + ["Ensemble"]
        )
        assert labels == expected_order
        plt.close(ax.figure)

    def test_order_alpha_sorts_by_name(self, boot_result_with_components):
        ax = forest(boot_result_with_components, order="alpha")
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["m0", "m1", "m2", "Ensemble"]
        plt.close(ax.figure)

    def test_accepts_existing_ax(self, boot_result_with_components):
        fig, ax = plt.subplots()
        ret = forest(boot_result_with_components, ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_raises_on_empty_components(self):
        boot = BootstrapResult(
            ate=0.0, ate_ci_lower=0.0, ate_ci_upper=0.0,
            boot_ates=np.zeros(1),
            cate=None, cate_ci_lower=None, cate_ci_upper=None,
        )
        with pytest.raises(ValueError, match="component_boot_ates is empty"):
            forest(boot)


# ---------------------------------------------------------------------------
# weights()
# ---------------------------------------------------------------------------


def _make_ew(values, names=None, method="mock"):
    values = np.asarray(values, dtype=float)
    if names is None:
        names = [f"m{i}" for i in range(len(values))]
    return EnsembleWeights(weights=values, model_names=list(names), method=method)


class TestWeights:
    def test_returns_axes_for_supervised_ensemble(self):
        ens, *_ = _fitted_supervised_ensemble(n=60, n_methods=3, seed=0)
        # UniformSupervisedStrategy produces uniform weights, so a
        # UserWarning is expected (and documented behaviour).
        with pytest.warns(UserWarning, match="uniform"):
            ax = weights(ens)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_accepts_ensemble_weights_directly(self):
        ew = _make_ew([0.5, 0.3, 0.2])
        ax = weights(ew)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_raises_for_pointwise_ensemble(self):
        # _fitted_supervised_ensemble uses a supervised strategy; build a
        # pointwise one instead.
        from metacausal.datasets import load_lalonde as _load
        # Use a minimal mock adapter + Median aggregator; avoid real fits.
        ens = CausalEnsemble(aggregation="median")
        X, T, Y = _load()
        with pytest.raises(ValueError, match="pointwise strategy"):
            # Not actually fitted — exercising the plot path only.
            weights(ens)

    def test_on_uniform_warn(self):
        ew = _make_ew([0.25, 0.25, 0.25, 0.25])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ax = weights(ew, on_uniform="warn")
            plt.close(ax.figure)
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_on_uniform_error(self):
        ew = _make_ew([0.25, 0.25, 0.25, 0.25])
        with pytest.raises(ValueError, match="uniform"):
            weights(ew, on_uniform="error")

    def test_on_uniform_ignore(self):
        ew = _make_ew([0.25, 0.25, 0.25, 0.25])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ax = weights(ew, on_uniform="ignore")
            plt.close(ax.figure)
        assert not any(issubclass(w.category, UserWarning) for w in caught)

    def test_sort_reorders_bars(self):
        ew = _make_ew([0.1, 0.7, 0.2], names=["a", "b", "c"])
        ax = weights(ew, sort=True)
        # barh draws bottom-up; our function reverses names so the largest
        # is at the top (last label in tick list).
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels[-1] == "b"
        plt.close(ax.figure)


# ---------------------------------------------------------------------------
# cate_profile()
# ---------------------------------------------------------------------------


def _make_cate_estimate(grid_size=10, k=3):
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.2, size=grid_size)
    comps = {
        f"m{i}": ComponentCateEstimate(
            cate=base + rng.normal(0, 0.1, size=grid_size)
        )
        for i in range(k)
    }
    return CateEstimate(
        cate=base, component_estimates=comps, aggregation="mock"
    )


def _make_boot_with_cate(grid_size=10):
    rng = np.random.default_rng(0)
    cate = rng.normal(0.5, 0.2, size=grid_size)
    return BootstrapResult(
        ate=0.5, ate_ci_lower=0.1, ate_ci_upper=0.9,
        boot_ates=rng.normal(0.5, 0.2, size=20),
        cate=cate,
        cate_ci_lower=cate - 0.2,
        cate_ci_upper=cate + 0.2,
    )


class TestCateProfile:
    def test_from_cate_estimate(self):
        est = _make_cate_estimate(grid_size=15, k=3)
        ax = cate_profile(est, np.arange(15), xlabel="x")
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_from_bootstrap_result(self):
        boot = _make_boot_with_cate(grid_size=15)
        ax = cate_profile(boot, np.arange(15), xlabel="x")
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_cate_estimate_has_no_ci_band(self):
        est = _make_cate_estimate(grid_size=10, k=3)
        ax = cate_profile(est, np.arange(10), xlabel="x")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert not any("CI" in lbl for lbl in labels)
        plt.close(ax.figure)

    def test_bootstrap_result_shows_ci_band(self):
        boot = _make_boot_with_cate(grid_size=10)
        ax = cate_profile(boot, np.arange(10), xlabel="x")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("CI" in lbl for lbl in labels)
        plt.close(ax.figure)

    def test_ylim_applied(self):
        est = _make_cate_estimate(grid_size=10, k=2)
        ax = cate_profile(est, np.arange(10), xlabel="x", ylim=(-5, 5))
        lo, hi = ax.get_ylim()
        assert lo == -5 and hi == 5
        plt.close(ax.figure)

    def test_raises_on_grid_length_mismatch(self):
        est = _make_cate_estimate(grid_size=10, k=2)
        with pytest.raises(ValueError, match="Grid length mismatch"):
            cate_profile(est, np.arange(5), xlabel="x")

    def test_raises_on_bootstrap_without_cate(self):
        boot = BootstrapResult(
            ate=0.0, ate_ci_lower=0.0, ate_ci_upper=0.0,
            boot_ates=np.zeros(1),
            cate=None, cate_ci_lower=None, cate_ci_upper=None,
        )
        with pytest.raises(ValueError, match="no CATE data"):
            cate_profile(boot, np.arange(5), xlabel="x")

    def test_bootstrap_result_draws_component_lines_when_available(self):
        """show_components=True overlays per-component CATE lines from
        BootstrapResult.component_cate_estimates when populated."""
        grid_size = 12
        rng = np.random.default_rng(0)
        cate = rng.normal(0.5, 0.2, size=grid_size)
        comps = {
            f"m{i}": ComponentCateEstimate(
                cate=cate + rng.normal(0, 0.1, size=grid_size)
            )
            for i in range(3)
        }
        boot = BootstrapResult(
            ate=0.5, ate_ci_lower=0.1, ate_ci_upper=0.9,
            boot_ates=rng.normal(0.5, 0.2, size=20),
            cate=cate,
            cate_ci_lower=cate - 0.2,
            cate_ci_upper=cate + 0.2,
            component_cate_estimates=comps,
        )
        x = np.arange(grid_size)
        ax_with = cate_profile(boot, x, xlabel="x", show_components=True)
        n_lines_with = len(ax_with.get_lines())
        plt.close(ax_with.figure)
        ax_without = cate_profile(boot, x, xlabel="x", show_components=False)
        n_lines_without = len(ax_without.get_lines())
        plt.close(ax_without.figure)
        assert n_lines_with - n_lines_without == 3


# ---------------------------------------------------------------------------
# disagreement()
# ---------------------------------------------------------------------------


class TestDisagreement:
    def test_returns_axes(self, cate_grid_ensemble):
        ens, X = cate_grid_ensemble
        ax = disagreement(ens, X)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    @pytest.mark.parametrize("metric", ["spearman", "pearson", "rmse"])
    def test_all_metrics(self, cate_grid_ensemble, metric):
        ens, X = cate_grid_ensemble
        ax = disagreement(ens, X, metric=metric)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_matrix_is_k_by_k_with_unit_diagonal(self, cate_grid_ensemble):
        ens, X = cate_grid_ensemble
        ax = disagreement(ens, X, metric="spearman")
        # The heatmap's image data should be KxK with ones on the diagonal.
        im = ax.images[0]
        matrix = im.get_array()
        assert matrix.shape[0] == matrix.shape[1]
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-8)
        plt.close(ax.figure)

    def test_rmse_diagonal_is_zero(self, cate_grid_ensemble):
        ens, X = cate_grid_ensemble
        ax = disagreement(ens, X, metric="rmse")
        matrix = ax.images[0].get_array()
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-8)
        plt.close(ax.figure)

    def test_cluster_reorders(self, cate_grid_ensemble):
        ens, X = cate_grid_ensemble
        ax = disagreement(ens, X, cluster=True)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_raises_with_fewer_than_two_components(self):
        ens, X, *_ = _fitted_supervised_ensemble(n=60, n_methods=1, seed=0)
        with pytest.raises(ValueError, match="at least 2"):
            disagreement(ens, X)


# ---------------------------------------------------------------------------
# Method delegates (editor Comment 5 — each plot also lives on its class)
# ---------------------------------------------------------------------------


class TestMethodDelegates:
    """``BootstrapResult.forest()``/``.cate_profile()``,
    ``CateEstimate.cate_profile()``, ``EnsembleWeights.plot()``, and
    ``CausalEnsemble.weights()``/``.disagreement()`` are thin wrappers
    around the corresponding ``metacausal.plots`` function. Each test
    checks the method and the function it wraps produce the same output
    on the same input.
    """

    def test_bootstrap_result_forest(self, boot_result_with_components):
        ax_method = boot_result_with_components.forest()
        labels_method = [t.get_text() for t in ax_method.get_yticklabels()]
        plt.close(ax_method.figure)

        ax_fn = forest(boot_result_with_components)
        labels_fn = [t.get_text() for t in ax_fn.get_yticklabels()]
        plt.close(ax_fn.figure)

        assert labels_method == labels_fn

    def test_bootstrap_result_cate_profile(self):
        boot = _make_boot_with_cate(grid_size=10)
        x = np.arange(10)

        ax_method = boot.cate_profile(x, xlabel="x")
        y_method = ax_method.get_lines()[0].get_ydata()
        plt.close(ax_method.figure)

        ax_fn = cate_profile(boot, x, xlabel="x")
        y_fn = ax_fn.get_lines()[0].get_ydata()
        plt.close(ax_fn.figure)

        np.testing.assert_allclose(y_method, y_fn)

    def test_cate_estimate_cate_profile(self):
        est = _make_cate_estimate(grid_size=10, k=3)
        x = np.arange(10)

        ax_method = est.cate_profile(x, xlabel="x")
        y_method = ax_method.get_lines()[0].get_ydata()
        plt.close(ax_method.figure)

        ax_fn = cate_profile(est, x, xlabel="x")
        y_fn = ax_fn.get_lines()[0].get_ydata()
        plt.close(ax_fn.figure)

        np.testing.assert_allclose(y_method, y_fn)

    def test_ensemble_weights_plot(self):
        ew = _make_ew([0.1, 0.7, 0.2], names=["a", "b", "c"])

        ax_method = ew.plot()
        labels_method = [t.get_text() for t in ax_method.get_yticklabels()]
        plt.close(ax_method.figure)

        ax_fn = weights(ew)
        labels_fn = [t.get_text() for t in ax_fn.get_yticklabels()]
        plt.close(ax_fn.figure)

        assert labels_method == labels_fn

    def test_causal_ensemble_weights(self):
        ens, *_ = _fitted_supervised_ensemble(n=60, n_methods=3, seed=0)

        with pytest.warns(UserWarning, match="uniform"):
            ax_method = ens.weights()
        plt.close(ax_method.figure)

        with pytest.warns(UserWarning, match="uniform"):
            ax_fn = weights(ens)
        plt.close(ax_fn.figure)

        assert isinstance(ax_method, plt.Axes)
        assert isinstance(ax_fn, plt.Axes)

    def test_causal_ensemble_disagreement(self, cate_grid_ensemble):
        ens, X = cate_grid_ensemble

        ax_method = ens.disagreement(X, metric="spearman")
        matrix_method = ax_method.images[0].get_array()
        plt.close(ax_method.figure)

        ax_fn = disagreement(ens, X, metric="spearman")
        matrix_fn = ax_fn.images[0].get_array()
        plt.close(ax_fn.figure)

        np.testing.assert_allclose(matrix_method, matrix_fn)
