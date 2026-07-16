"""AggregationStrategy ABC, PointwiseStrategy, AgreementStrategy, and SupervisedStrategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from metacausal.aggregation.splitting import CrossFitSplit, TrainAvgSplit
from metacausal.aggregation.weights import EnsembleWeights


class AggregationStrategy(ABC):
    """Abstract base class for all aggregation strategies.

    Three concrete families inherit from this base — :class:`PointwiseStrategy`,
    :class:`AgreementStrategy`, :class:`SupervisedStrategy` — and runtime
    dispatch in :class:`CausalEnsemble` keys off the family base, since each
    family has a different fit-time and predict-time contract. All three
    families implement the same prediction-time operation: :meth:`aggregate`
    reduces along axis 0.
    """

    @abstractmethod
    def aggregate(self, values: np.ndarray) -> np.ndarray:
        """Reduce a ``(K, n)`` component-CATE matrix to a ``(n,)`` ensemble CATE.

        ``K`` is the number of component models; ``n`` is the number of
        evaluation points. Subclasses define the per-family rule (pointwise
        statistical reduction, weighted combination, learned linear
        combination, etc.).

        :class:`PointwiseStrategy` extends this contract to also accept
        ``(K,)`` 1-D input (returning a 0-d scalar) for the component-ATE
        aggregation path; see its class docstring. Other families do not
        meaningfully support 1-D input.
        """

    @property
    @abstractmethod
    def ensemble_weights(self) -> EnsembleWeights | None:
        """Ensemble weights, if applicable. ``None`` for pointwise strategies."""


@dataclass
class PointwiseStrategy(AggregationStrategy):
    """Base class for strategies that apply a fixed statistical rule pointwise.

    Subclasses implement a single :meth:`aggregate` method that reduces along
    axis 0 using NumPy ``axis=0``-aware operations (``np.median``, ``np.mean``,
    etc.). Because those operations are rank-agnostic, the same implementation
    handles both the CATE path (``(K, n)`` → ``(n,)``, the base contract) and
    the ATE-from-component-scalars path (``(K,)`` → 0-d scalar) — the latter
    is a Pointwise-specific extension of the base ``aggregate`` contract.
    Pointwise strategies are stateless: no weights are computed at fit time.
    """

    strategy_family: ClassVar[str] = "pointwise"

    @property
    def ensemble_weights(self) -> None:
        return None


@dataclass
class AgreementStrategy(AggregationStrategy):
    """Base class for strategies that select/weight models by inter-model agreement.

    Weights are computed at fit time from training-data CATE predictions
    (no outcome data involved). Subclasses implement :meth:`compute_weights`.

    ATE is computed as the mean of the ensemble CATE, not by aggregating
    scalar component ATEs. For bootstrap replicates, weights are recomputed
    from resampled training-data CATE predictions.
    """

    strategy_family: ClassVar[str] = "agreement"

    _weights: EnsembleWeights | None = field(default=None, init=False, repr=False)

    @abstractmethod
    def compute_weights(
        self,
        cate_matrix: np.ndarray,
        model_names: list[str],
    ) -> EnsembleWeights:
        """Compute weights from inter-model agreement on training predictions.

        Called once during ``fit()``, after component models are fitted on full
        training data. Subclasses implement this.

        Parameters
        ----------
        cate_matrix : array of shape (K, n)
            Training-data CATE predictions, one row per model.
        model_names : list[str]
            Adapter names in the same order as rows of ``cate_matrix``.

        Returns
        -------
        EnsembleWeights populated with this strategy's weights.
        """

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        """Aggregate component CATEs using stored weights.

        Parameters
        ----------
        values : array of shape (K, n)

        Returns
        -------
        Array of shape (n,).
        """
        w = self._weights
        return w.intercept + w.weights @ values

    @property
    def ensemble_weights(self) -> EnsembleWeights | None:
        """Ensemble weights computed during ``fit()``. ``None`` before ``fit()``."""
        return self._weights


@dataclass
class SupervisedStrategy(AggregationStrategy):
    """Base class for outcome-supervised aggregation strategies.

    Weights are learned by optimizing a causal loss on outcome data, using
    out-of-fold CATE predictions and nuisance model estimates. Subclasses
    implement :meth:`fit_weights`.

    ATE is computed as the mean of the ensemble CATE (same as
    :class:`AgreementStrategy`). ATE-only adapters (``supports_cate=False``)
    are excluded from both CATE and ATE computation when this strategy is
    active.

    Parameters
    ----------
    split : CrossFitSplit or TrainAvgSplit
        Data splitting strategy for cross-fitting. Default: 5-fold cross-fitting.
    propensity_model : sklearn classifier or None
        Model for P(T=1|X). Default: HistGradientBoostingClassifier (from defaults.py).
    outcome_model : sklearn regressor / classifier or None
        Model for E[Y|X,T]. For continuous Y, must be a regressor; for binary
        Y, must be a classifier with ``predict_proba``. ``None`` (default)
        selects an outcome-type-appropriate default at fit time.
    propensity_trim : float
        Clip propensity scores to [trim, 1-trim] to enforce overlap.
    fit_nuisance_fn : callable or None
        Optional replacement for the default ``fit_nuisance`` function.
        Must have the same signature::

            fit_nuisance_fn(X, T, Y, fold_spec,
                            propensity_model, outcome_model,
                            propensity_trim, random_state,
                            outcome_type) -> NuisanceEstimates

        ``outcome_type`` is the resolved value (``"continuous"`` or
        ``"binary"``) the ensemble has decided for this fit; respect it when
        choosing whether to call ``predict()`` or ``predict_proba()``.

        Use this to inject BART-based nuisance, a single pooled outcome model
        with T as a feature, or any other custom nuisance pipeline.
        ``None`` (default) uses the standard two-model cross-fitted procedure.
    """

    strategy_family: ClassVar[str] = "supervised"

    split: CrossFitSplit | TrainAvgSplit = field(default_factory=CrossFitSplit)
    propensity_model: Any = None
    outcome_model: Any = None
    propensity_trim: float = 0.01
    fit_nuisance_fn: Any = field(default=None, repr=False)

    _weights: EnsembleWeights | None = field(default=None, init=False, repr=False)

    @abstractmethod
    def fit_weights(
        self,
        cate_predictions: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        nuisance: Any,  # NuisanceEstimates — avoid importing nuisance.py here
    ) -> EnsembleWeights:
        """Compute ensemble weights from OOF predictions and nuisance estimates.

        Parameters
        ----------
        cate_predictions : array of shape (K, m)
            Out-of-fold CATE predictions. m = n for CrossFitSplit;
            m = len(averaging set) for TrainAvgSplit.
        Y, T, X : arrays of shape (m,), (m,), (m, p)
            Outcome, treatment, and covariates for the same m observations.
        nuisance : NuisanceEstimates
            Out-of-fold nuisance predictions for the same m observations.

        Returns
        -------
        EnsembleWeights with optimized weights.
        """

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        """Aggregate component CATEs using stored weights.

        Parameters
        ----------
        values : array of shape (K, n)

        Returns
        -------
        Array of shape (n,).
        """
        w = self._weights
        return w.intercept + w.weights @ values

    @property
    def ensemble_weights(self) -> EnsembleWeights | None:
        """Ensemble weights computed during ``fit()``. ``None`` before ``fit()``."""
        return self._weights
