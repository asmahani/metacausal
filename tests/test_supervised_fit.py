"""Tests for the supervised fit path (_fit_supervised) in CausalEnsemble."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from metacausal import (
    CausalEnsemble,
    ComponentExclusionWarning,
    ComponentFailureWarning,
)
from metacausal.aggregation import CrossFitSplit, SupervisedStrategy, TrainAvgSplit
from metacausal.aggregation.weights import EnsembleWeights
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


# ---------------------------------------------------------------------------
# Mock supervised strategy
# ---------------------------------------------------------------------------


@dataclass
class UniformSupervisedStrategy(SupervisedStrategy):
    """Supervised strategy that assigns uniform weights. Used for testing."""

    def fit_weights(self, cate_predictions, Y, T, X, nuisance) -> EnsembleWeights:
        K = cate_predictions.shape[0]
        weights = np.ones(K) / K
        self._weights = EnsembleWeights(
            weights=weights,
            model_names=[],  # _fit_supervised fills this in after the call
            method="uniform_mock",
        )
        return self._weights


def _make_mock_strategy(n_folds: int = 3) -> UniformSupervisedStrategy:
    return UniformSupervisedStrategy(
        split=CrossFitSplit(n_folds=n_folds, stratify=False),
        propensity_model=LogisticRegression(max_iter=200),
        outcome_model=LinearRegression(),
    )


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------


class CateCapableAdapter:
    """Minimal CATE-capable adapter for testing."""

    def __init__(self, name: str, scale: float = 1.0):
        self._name = name
        self._scale = scale
        self._fitted = False
        self._X_train: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, random_state=None, **kwargs):
        self._X_train = X.copy()
        self._fitted = True

    def ate(self, X=None):
        n = len(X) if X is not None else len(self._X_train)
        return ComponentAteEstimate(ate=float(self._scale * np.ones(n).mean()))

    def cate(self, X):
        return ComponentCateEstimate(
            cate=self._scale * np.ones(X.shape[0])
        )

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class AteOnlyAdapter:
    """Minimal ATE-only adapter (supports_cate=False) for testing.

    Has a cate() stub as required by the CausalEstimator protocol, but
    signals via supports_cate=False that it should not be used for CATE.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def fit(self, X, T, Y, random_state=None, **kwargs):
        pass

    def ate(self, X=None):
        return ComponentAteEstimate(ate=0.5)

    def cate(self, X):
        raise NotImplementedError("This adapter does not support CATE.")

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


class FailingInFoldAdapter:
    """Adapter that raises during fit() only when given small training data."""

    def __init__(self, name: str, fail_threshold: int = 50):
        self._name = name
        self._fail_threshold = fail_threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def fit(self, X, T, Y, random_state=None, **kwargs):
        if len(X) < self._fail_threshold:
            raise ValueError(f"Intentional failure on small data (n={len(X)})")

    def ate(self, X=None):
        return ComponentAteEstimate(ate=0.0)

    def cate(self, X):
        return ComponentCateEstimate(cate=np.zeros(X.shape[0]))

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    def validate_outcome_type(self, detected: str) -> None:
        pass


# ---------------------------------------------------------------------------
# DGP helper
# ---------------------------------------------------------------------------


def _dgp(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 0.5 * np.ones(n)
    T = rng.binomial(1, e).astype(float)
    Y = X[:, 0] + T * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, T, Y


# ---------------------------------------------------------------------------
# Basic dispatch and fit
# ---------------------------------------------------------------------------


class TestSupervisedDispatch:
    def test_fit_runs_without_error(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        assert ens._is_fitted

    def test_fitted_adapters_only_cate_capable(self):
        """After supervised fit, _fitted_adapters contains only CATE-capable adapters."""
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[
                CateCapableAdapter("cate1"),
                AteOnlyAdapter("ate_only"),
                CateCapableAdapter("cate2"),
            ],
            aggregation=_make_mock_strategy(),
        )
        with pytest.warns(ComponentExclusionWarning, match="ATE-only adapters will be skipped"):
            ens.fit(X, T, Y, random_state=0)

        names = [a.name for a in ens._fitted_adapters]
        assert "cate1" in names
        assert "cate2" in names
        assert "ate_only" not in names

    def test_warning_for_ate_only_adapters(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("c"), AteOnlyAdapter("a")],
            aggregation=_make_mock_strategy(),
        )
        with pytest.warns(ComponentExclusionWarning, match="ATE-only adapters will be skipped"):
            ens.fit(X, T, Y, random_state=0)

    def test_no_warning_without_ate_only_adapters(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ens.fit(X, T, Y, random_state=0)
        skipped_warnings = [w for w in caught if "ATE-only" in str(w.message)]
        assert len(skipped_warnings) == 0

    def test_no_cate_methods_raises(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[AteOnlyAdapter("a"), AteOnlyAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        with pytest.raises(RuntimeError, match="at least one CATE-capable"):
            ens.fit(X, T, Y, random_state=0)


# ---------------------------------------------------------------------------
# Ensemble weights
# ---------------------------------------------------------------------------


class TestSupervisedWeights:
    def test_ensemble_weights_populated(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        w = ens.aggregation.ensemble_weights
        assert w is not None
        assert isinstance(w, EnsembleWeights)

    def test_model_names_set_on_weights(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        names = ens.aggregation.ensemble_weights.model_names
        assert set(names) == {"a", "b"}

    def test_uniform_weights_sum_to_one(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b"), CateCapableAdapter("c")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        w = ens.aggregation.ensemble_weights.weights
        np.testing.assert_allclose(w.sum(), 1.0)
        assert len(w) == 3

    def test_cate_ensemble_weights_populated(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.ensemble_weights is not None


# ---------------------------------------------------------------------------
# cate() and ate() after supervised fit
# ---------------------------------------------------------------------------


class TestSupervisedPrediction:
    def test_cate_shape(self):
        n = 200
        X, T, Y = _dgp(n=n)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", scale=1.0), CateCapableAdapter("b", scale=2.0)],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        assert result.cate.shape == (n,)

    def test_ate_equals_mean_of_cate(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", scale=1.0), CateCapableAdapter("b", scale=3.0)],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        ate_result = ens.ate(X)
        cate_result = ens.cate(X)
        np.testing.assert_allclose(ate_result.ate, cate_result.cate.mean())

    def test_uniform_weights_average_components(self):
        """With uniform weights, ensemble CATE = mean of component CATEs."""
        X, T, Y = _dgp(n=100)
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a", scale=1.0), CateCapableAdapter("b", scale=3.0)],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X)
        # a produces 1.0, b produces 3.0 → uniform mean = 2.0
        np.testing.assert_allclose(result.cate, 2.0 * np.ones(100), rtol=1e-6)

    def test_cate_on_new_X(self):
        X, T, Y = _dgp(n=200)
        X_new = np.random.default_rng(99).normal(size=(50, 3))
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        result = ens.cate(X_new)
        assert result.cate.shape == (50,)


# ---------------------------------------------------------------------------
# OOF correctness
# ---------------------------------------------------------------------------


class TestOOFCorrectness:
    def test_oof_predictions_out_of_sample(self):
        """OOF predictions differ from in-sample (full-data) predictions.

        We use an adapter that memorizes its training data and returns
        a distinctive prediction for in-sample vs out-of-sample observations.
        """

        class MemorizingAdapter:
            """Returns 1.0 for observations seen during training, 0.0 otherwise."""

            def __init__(self, name: str):
                self._name = name
                self._train_set: frozenset | None = None

            @property
            def name(self):
                return self._name

            @property
            def supports_cate(self):
                return True

            def fit(self, X, T, Y, random_state=None, **kwargs):
                # Store a hash of each row to identify training observations
                self._train_set = frozenset(map(tuple, X.round(8)))

            def ate(self, X=None):
                return ComponentAteEstimate(ate=0.0)

            def cate(self, X):
                preds = np.array([
                    1.0 if tuple(row.round(8)) in self._train_set else 0.0
                    for row in X
                ])
                return ComponentCateEstimate(cate=preds)

            supported_outcome_types: tuple[str, ...] = ("continuous",)

            def validate_outcome_type(self, detected: str) -> None:
                pass

        n = 100
        X, T, Y = _dgp(n=n, seed=7)
        strategy = UniformSupervisedStrategy(
            split=CrossFitSplit(n_folds=5, stratify=False),
            propensity_model=LogisticRegression(max_iter=200),
            outcome_model=LinearRegression(),
        )
        ens = CausalEnsemble(
            methods=[MemorizingAdapter("mem")],
            aggregation=strategy,
        )
        ens.fit(X, T, Y, random_state=0)

        # OOF predictions: for each test fold, the adapter was NOT trained
        # on those observations, so it returns 0.0 (not recognized).
        oof = ens._cached_oof_cate_predictions  # shape (1, n)
        assert oof is not None
        # The OOF predictions on test observations should be 0.0
        # (not in training set for that fold), not 1.0 (in-sample).
        assert np.all(oof[0] == 0.0), (
            "OOF predictions should be 0.0 (out-of-sample), got in-sample (1.0) values"
        )

    def test_oof_artifacts_cached(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        assert ens._cached_oof_cate_predictions is not None
        assert ens._cached_oof_cate_model_names is not None
        assert ens._cached_nuisance is not None

    def test_oof_model_names_match_weights(self):
        X, T, Y = _dgp()
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=_make_mock_strategy(),
        )
        ens.fit(X, T, Y, random_state=0)
        assert ens._cached_oof_cate_model_names == ens.aggregation.ensemble_weights.model_names


# ---------------------------------------------------------------------------
# Adapter failure handling
# ---------------------------------------------------------------------------


class TestAdapterFailureInCrossFitting:
    def test_failed_adapter_excluded_from_all_folds(self):
        """An adapter that fails in any fold is excluded from _fitted_adapters."""
        n = 200
        X, T, Y = _dgp(n=n)
        # FailingInFoldAdapter fails when training set < 50 obs.
        # With n=200 and 3 folds, each training fold has ~133 obs — no failure.
        # Force failure by setting a high threshold.
        ens = CausalEnsemble(
            methods=[
                CateCapableAdapter("good"),
                FailingInFoldAdapter("bad", fail_threshold=n),  # always fails
            ],
            aggregation=_make_mock_strategy(n_folds=3),
        )
        with pytest.warns(ComponentFailureWarning, match="Dropping from all folds"):
            ens.fit(X, T, Y, random_state=0)

        names = [a.name for a in ens._fitted_adapters]
        assert "good" in names
        assert "bad" not in names

    def test_failed_adapter_not_in_weights(self):
        n = 200
        X, T, Y = _dgp(n=n)
        ens = CausalEnsemble(
            methods=[
                CateCapableAdapter("good"),
                FailingInFoldAdapter("bad", fail_threshold=n),
            ],
            aggregation=_make_mock_strategy(n_folds=3),
        )
        with pytest.warns(ComponentFailureWarning, match="Dropping from all folds"):
            ens.fit(X, T, Y, random_state=0)

        assert "bad" not in ens.aggregation.ensemble_weights.model_names

    def test_all_adapters_fail_raises(self):
        n = 200
        X, T, Y = _dgp(n=n)
        ens = CausalEnsemble(
            methods=[
                FailingInFoldAdapter("a", fail_threshold=n),
                FailingInFoldAdapter("b", fail_threshold=n),
            ],
            aggregation=_make_mock_strategy(n_folds=3),
        )
        with pytest.raises(RuntimeError, match="Supervised fit pipeline failed"):
            with pytest.warns(ComponentFailureWarning):
                ens.fit(X, T, Y, random_state=0)


# ---------------------------------------------------------------------------
# TrainAvgSplit
# ---------------------------------------------------------------------------


class TestTrainAvgSplitPath:
    def test_fit_with_train_avg_split(self):
        X, T, Y = _dgp(n=200)
        strategy = UniformSupervisedStrategy(
            split=TrainAvgSplit(avg_frac=0.25, stratify=False),
            propensity_model=LogisticRegression(max_iter=200),
            outcome_model=LinearRegression(),
        )
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=strategy,
        )
        ens.fit(X, T, Y, random_state=0)
        assert ens._is_fitted
        assert ens.aggregation.ensemble_weights is not None

    def test_oof_predictions_are_avg_set_only(self):
        """With TrainAvgSplit, OOF predictions have shape (K, avg_n), not (K, n)."""
        n = 200
        avg_frac = 0.25
        X, T, Y = _dgp(n=n)
        strategy = UniformSupervisedStrategy(
            split=TrainAvgSplit(avg_frac=avg_frac, stratify=False),
            propensity_model=LogisticRegression(max_iter=200),
            outcome_model=LinearRegression(),
        )
        ens = CausalEnsemble(
            methods=[CateCapableAdapter("a"), CateCapableAdapter("b")],
            aggregation=strategy,
        )
        ens.fit(X, T, Y, random_state=0)
        oof = ens._cached_oof_cate_predictions
        expected_avg_n = int(n * avg_frac)
        # Allow ±1 for rounding in train_test_split
        assert abs(oof.shape[1] - expected_avg_n) <= 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_weights(self):
        X, T, Y = _dgp(n=200)
        methods = [CateCapableAdapter("a"), CateCapableAdapter("b")]

        ens1 = CausalEnsemble(methods=methods, aggregation=_make_mock_strategy())
        ens2 = CausalEnsemble(methods=methods, aggregation=_make_mock_strategy())
        ens1.fit(X, T, Y, random_state=42)
        ens2.fit(X, T, Y, random_state=42)

        np.testing.assert_array_equal(
            ens1.aggregation.ensemble_weights.weights,
            ens2.aggregation.ensemble_weights.weights,
        )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_supervised_strategy_importable():
    from metacausal.aggregation import SupervisedStrategy  # noqa: F401
