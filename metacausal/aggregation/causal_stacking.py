"""Causal Stacking (Han & Wu, 2022)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from metacausal.aggregation.base import SupervisedStrategy
from metacausal.aggregation.q_aggregation import _solve_dr_simplex
from metacausal.aggregation.weights import EnsembleWeights


@dataclass
class CausalStacking(SupervisedStrategy):
    """Causal Stacking (Han & Wu, 2022).

    Finds a convex combination of component CATEs that minimises the mean
    squared error of the ensemble prediction against the DR/AIPW
    pseudo-outcome:

        min_{θ ≥ 0, Σθ=1}  (1/m) Σ_i (Γ_i - θ·τ(x_i))²

    This is the special case of Q-Aggregation with ν=0 and β=0.

    Parameters
    ----------
    split : CrossFitSplit or TrainAvgSplit
        Data splitting strategy. Default: 5-fold cross-fitting.
    propensity_model : sklearn classifier or None
        Model for P(T=1|X). ``None`` selects ``HistGradientBoostingClassifier``.
    outcome_model : sklearn regressor / classifier or None
        Model for E[Y|X, T]. For continuous Y, must be a regressor; for binary
        Y, must be a classifier (``predict_proba``). ``None`` selects an
        outcome-type-appropriate default (``HistGradientBoostingRegressor`` or
        ``HistGradientBoostingClassifier``).
    propensity_trim : float
        Clip propensity scores to [trim, 1-trim].
    fit_nuisance_fn : callable or None
        Optional replacement for the default nuisance fitting procedure.
        See ``SupervisedStrategy.fit_nuisance_fn`` for the required signature.
    pseudo_outcome_fn : callable or None
        Optional replacement for ``dr_pseudo_outcome``. Must have signature::

            pseudo_outcome_fn(Y, T, nuisance) -> ndarray of shape (m,)

        Use this to inject targeted-learning influence functions or other
        semiparametric pseudo-outcome constructions.
        ``None`` (default) uses the standard AIPW form.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    >>> from metacausal import CausalEnsemble
    >>> from metacausal.adapters import GenericCATEAdapter
    >>> from metacausal.aggregation import CausalStacking
    >>> from metacausal.datasets import load_lalonde
    >>> X, T, Y = load_lalonde()
    >>> def fit_linear(X, T, Y, **kwargs):
    ...     treated = T == 1
    ...     m1 = LinearRegression().fit(X[treated], Y[treated])
    ...     m0 = LinearRegression().fit(X[~treated], Y[~treated])
    ...     return (m1, m0)
    >>> def fit_hgb(X, T, Y, **kwargs):
    ...     treated = T == 1
    ...     m1 = HGBR(max_iter=20).fit(X[treated], Y[treated])
    ...     m0 = HGBR(max_iter=20).fit(X[~treated], Y[~treated])
    ...     return (m1, m0)
    >>> def cate_fn(state, X):
    ...     m1, m0 = state
    ...     return m1.predict(X) - m0.predict(X)
    >>> methods = [
    ...     GenericCATEAdapter(fit_linear, cate_fn, name="linear"),
    ...     GenericCATEAdapter(fit_hgb, cate_fn, name="hgb"),
    ... ]
    >>> ens = CausalEnsemble(methods=methods, aggregation=CausalStacking())
    >>> _ = ens.fit(X, T, Y, random_state=42)
    >>> ens.ate()
    AteEstimate(ate=..., n_methods=2, aggregation='CausalStacking', spread=...)
    """

    pseudo_outcome_fn: Any = field(default=None, repr=False)

    def fit_weights(
        self,
        cate_predictions: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        nuisance: Any,
    ) -> EnsembleWeights:
        """Compute Causal Stacking weights from OOF predictions and nuisance estimates.

        Parameters
        ----------
        cate_predictions : array of shape (K, m)
        Y, T : arrays of shape (m,)
        X : array of shape (m, p)
        nuisance : NuisanceEstimates

        Returns
        -------
        EnsembleWeights (model_names populated by _fit_supervised).
        """
        from metacausal.aggregation.nuisance import dr_pseudo_outcome

        K = cate_predictions.shape[0]

        if K == 1:
            self._weights = EnsembleWeights(
                weights=np.array([1.0]),
                model_names=[],
                method="causal_stacking",
            )
            return self._weights

        pseudo_outcome_fn = self.pseudo_outcome_fn if self.pseudo_outcome_fn is not None else dr_pseudo_outcome
        pseudo_outcomes = pseudo_outcome_fn(Y, T, nuisance)

        weights = _solve_dr_simplex(
            pseudo_outcomes=pseudo_outcomes,
            cate_predictions=cate_predictions,
            nu=0.0,
            beta=0.0,
            prior=np.ones(K),
            greedy=False,
        )

        self._weights = EnsembleWeights(
            weights=weights,
            model_names=[],
            method="causal_stacking",
        )
        return self._weights
