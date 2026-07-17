"""Select: feasible best-member selection (Alaa & van der Schaar, 2019; Nie & Wager, 2021)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from metacausal.aggregation.base import SupervisedStrategy
from metacausal.aggregation.weights import EnsembleWeights


@dataclass
class Select(SupervisedStrategy):
    """Select the single component model minimizing a pseudo-outcome risk.

    Not an ensemble — a feasible selection procedure expressed as a
    :class:`SupervisedStrategy` so it shares the same fit-time contract
    (cross-fitted nuisances, out-of-fold CATE predictions) and downstream
    machinery (bootstrap, ``EnsembleWeights`` introspection) as the true
    aggregators. It is the natural comparator for "does aggregation beat
    feasible selection?" experiments.

    Two risk criteria:

    ``loss="dr"``
        DR/AIPW plug-in risk, ``mean_i (Gamma_i - tau_hat_k(x_i))**2``. At
        ``nu=1``, :class:`QAggregation`'s objective is exactly the weighted
        average of these per-model MSEs (see its docstring), which is linear
        in the simplex weights — so the constrained optimum sits at the
        one-hot vertex minimizing individual MSE. ``Select(loss="dr")``
        computes that vertex directly by argmin rather than relying on a QP
        solver to find it, and is in the lineage of influence-function-
        corrected plug-in validation (Alaa & van der Schaar, 2019).

    ``loss="r"``
        R-risk, ``mean_i (Y_tilde_i - tau_hat_k(x_i)*W_tilde_i)**2``, using
        the Robinson decomposition residuals (Nie & Wager, 2021; the
        selection criterion advocated by Schuler et al., 2018 and
        Doutreligne & Varoquaux, 2025).

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
    loss : str
        Risk criterion: ``"dr"`` (default) or ``"r"``.
    pseudo_outcome_fn : callable or None
        Optional replacement for ``dr_pseudo_outcome``, used only when
        ``loss="dr"``. Must have signature::

            pseudo_outcome_fn(Y, T, nuisance) -> ndarray of shape (m,)

        ``None`` (default) uses the standard AIPW form.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    >>> from metacausal import CausalEnsemble
    >>> from metacausal.adapters import GenericCATEAdapter
    >>> from metacausal.aggregation import Select
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
    >>> ens = CausalEnsemble(methods=methods, aggregation=Select(loss="dr"))
    >>> _ = ens.fit(X, T, Y, random_state=42)
    >>> ens.ate()
    AteEstimate(ate=..., n_methods=2, aggregation='Select', spread=...)
    """

    loss: str = "dr"
    pseudo_outcome_fn: Any = field(default=None, repr=False)

    def fit_weights(
        self,
        cate_predictions: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        nuisance: Any,
    ) -> EnsembleWeights:
        """Select the lowest-risk component from OOF predictions and nuisance estimates.

        Parameters
        ----------
        cate_predictions : array of shape (K, m)
        Y, T : arrays of shape (m,)
        X : array of shape (m, p)
        nuisance : NuisanceEstimates

        Returns
        -------
        EnsembleWeights with a one-hot ``weights`` vector at the selected
        model (model_names populated by ``_fit_supervised``); ``details``
        carries the full per-model risk vector.
        """
        K = cate_predictions.shape[0]

        if K == 1:
            self._weights = EnsembleWeights(
                weights=np.array([1.0]),
                model_names=[],
                method=f"select_{self.loss}",
            )
            return self._weights

        if self.loss == "dr":
            from metacausal.aggregation.nuisance import dr_pseudo_outcome

            pseudo_outcome_fn = (
                self.pseudo_outcome_fn if self.pseudo_outcome_fn is not None else dr_pseudo_outcome
            )
            gamma = pseudo_outcome_fn(Y, T, nuisance)
            risk = np.mean((gamma[np.newaxis, :] - cate_predictions) ** 2, axis=1)
        elif self.loss == "r":
            from metacausal.aggregation.nuisance import robinson_residuals

            Y_tilde, W_tilde = robinson_residuals(Y, T, nuisance)
            risk = np.mean(
                (Y_tilde[np.newaxis, :] - cate_predictions * W_tilde[np.newaxis, :]) ** 2,
                axis=1,
            )
        else:
            raise ValueError(f"loss must be 'dr' or 'r', got {self.loss!r}")

        j_star = int(np.argmin(risk))
        weights = np.zeros(K)
        weights[j_star] = 1.0

        self._weights = EnsembleWeights(
            weights=weights,
            model_names=[],
            method=f"select_{self.loss}",
            details={"risk": risk},
        )
        return self._weights
