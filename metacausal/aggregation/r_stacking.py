"""R-Stacking (Nie & Wager, 2021)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from metacausal.aggregation.base import SupervisedStrategy
from metacausal.aggregation.weights import EnsembleWeights


@dataclass
class RStacking(SupervisedStrategy):
    """R-Stacking (Nie & Wager, 2021).

    Finds a conic combination of component CATEs (plus a constant shift) that
    minimises the R-loss — the squared residual of the Robinson decomposition:

        min_{b, c, α ≥ 0}  Σ_i [Ỹ_i - b - (c + α·τ̂(X_i))·W̃_i]²

    where Ỹ_i = Y_i - m̂(X_i) and W̃_i = W_i - ê(X_i) are Robinson residuals.

    Reformulated as bounded linear least squares:

        min  ‖Az - y‖²   s.t.  lb ≤ z ≤ ub

        A : (n, K+2) design matrix
            col 0      : ones          → z[0] = b  (unconstrained)
            col 1      : W̃            → z[1] = c  (unconstrained)
            cols 2..K+1: τ̂_k·W̃       → z[2:] = α (α_k ≥ 0)

        y : Ỹ  (n,)

    Solved via ``scipy.optimize.lsq_linear``, which handles per-variable
    bounds directly.

    **Output:** ``EnsembleWeights(weights=α̂, intercept=ĉ)``.
    ``b̂`` absorbs residual mean bias in m̂ and is discarded — it does not
    appear in CATE prediction. ``ĉ`` is the estimated constant CATE shift;
    it becomes the ``intercept`` field of ``EnsembleWeights`` so that
    ``aggregate`` correctly computes ``ĉ + α̂·τ̂(x)``.

    **Key differences from CausalStacking / Q-Aggregation:**

    * **Conic** constraint (α_k ≥ 0, no sum-to-one): weights are not
      probabilities. The ensemble prediction is not a convex combination of
      individual CATEs but can amplify or dampen any component.
    * **Constant shift** ĉ: the ensemble adds a global CATE offset, enabling
      it to correct for systematic under- or over-estimation by the component
      library.
    * **R-loss** rather than DR pseudo-outcome MSE: requires both m̂ and ê
      (not separately doubly robust).

    **Numerical note:** When W̃_i ≈ 0 (highly accurate propensity model),
    the columns τ̂_k·W̃ are near-zero, providing little signal about the
    CATE. The solver may return α ≈ 0 with a large b. This is a fundamental
    limitation of R-loss in near-balanced settings, not an implementation bug.

    Parameters
    ----------
    split : CrossFitSplit or TrainAvgSplit
        Data splitting strategy. Default: 5-fold cross-fitting.
    propensity_model : sklearn classifier or None
        Model for P(T=1|X). ``None`` selects ``HistGradientBoostingClassifier``.
    outcome_model : sklearn regressor / classifier or None
        Model for E[Y|X, T]. The marginal conditional mean m̂(x) is derived
        from this via m̂ = ê·μ̂₁ + (1-ê)·μ̂₀. For continuous Y, must be a
        regressor; for binary Y, must be a classifier (``predict_proba``).
        ``None`` selects an outcome-type-appropriate default
        (``HistGradientBoostingRegressor`` or ``HistGradientBoostingClassifier``).
    propensity_trim : float
        Clip propensity scores to [trim, 1-trim].

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    >>> from metacausal import CausalEnsemble
    >>> from metacausal.adapters import GenericCATEAdapter
    >>> from metacausal.aggregation import RStacking
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
    >>> ens = CausalEnsemble(methods=methods, aggregation=RStacking())
    >>> _ = ens.fit(X, T, Y, random_state=42)
    >>> ens.ate()
    AteEstimate(ate=..., n_methods=2, aggregation='RStacking', spread=...)
    """

    def fit_weights(
        self,
        cate_predictions: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        nuisance: Any,
    ) -> EnsembleWeights:
        """Compute R-Stacking weights from OOF predictions and nuisance estimates.

        Parameters
        ----------
        cate_predictions : array of shape (K, m)
            Out-of-fold CATE predictions; one row per component model.
        Y, T : arrays of shape (m,)
            Observed outcome and treatment for the m OOF/averaging observations.
        X : array of shape (m, p)
            Covariates (not used directly; included for interface consistency).
        nuisance : NuisanceEstimates
            Out-of-fold nuisance predictions for the same m observations.
            Uses ``nuisance.m_hat`` and ``nuisance.e_hat`` for Robinson residuals.

        Returns
        -------
        EnsembleWeights with ``weights = α̂`` (shape K, non-negative) and
        ``intercept = ĉ``. Model names are populated by ``_fit_supervised``.
        """
        from metacausal.aggregation.nuisance import robinson_residuals

        K, m = cate_predictions.shape
        Y_tilde, W_tilde = robinson_residuals(Y, T, nuisance)

        # Design matrix A: (m, K+2)
        # col 0: intercept term b
        # col 1: W_tilde for constant shift c
        # cols 2..K+1: tau_k * W_tilde for each component model k
        A = np.column_stack([
            np.ones(m),                                           # b column
            W_tilde,                                              # c column
            cate_predictions.T * W_tilde[:, np.newaxis],         # alpha columns
        ])  # shape (m, K+2)

        # Bounds: b and c unconstrained, alpha_k >= 0
        lb = np.full(K + 2, -np.inf)
        lb[2:] = 0.0
        ub = np.full(K + 2, np.inf)

        result = lsq_linear(A, Y_tilde, bounds=(lb, ub), method="bvls")

        b_hat = result.x[0]   # absorbed mean bias — discarded
        c_hat = float(result.x[1])
        alpha_hat = result.x[2:]

        # Clip tiny negatives from numerical noise (bounds guarantee >= 0
        # but floating-point can leave values like -1e-14)
        alpha_hat = np.maximum(alpha_hat, 0.0)

        self._weights = EnsembleWeights(
            weights=alpha_hat,
            model_names=[],        # populated by _fit_supervised
            intercept=c_hat,
            method="r_stacking",
            details={"b_hat": float(b_hat)},
        )
        return self._weights
