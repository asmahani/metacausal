"""Q-Aggregation (Lan & Syrgkanis, 2024) and shared DR simplex solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from metacausal.aggregation.base import SupervisedStrategy
from metacausal.aggregation.splitting import CrossFitSplit, TrainAvgSplit
from metacausal.aggregation.weights import EnsembleWeights


# ---------------------------------------------------------------------------
# Shared DR simplex solver
# ---------------------------------------------------------------------------


def _solve_dr_simplex(
    pseudo_outcomes: np.ndarray,
    cate_predictions: np.ndarray,
    nu: float,
    beta: float,
    prior: np.ndarray,
    greedy: bool,
) -> np.ndarray:
    """Solve the DR simplex weight optimisation.

    Minimises the Q-aggregation objective over the probability simplex:

        L(θ) = (1/m) Σ_i [(1-ν)(Γ_i - θ·τ_i)² + ν Σ_k θ_k (Γ_i - τ_k(x_i))²]
               + (β/m) Σ_k θ_k log(1/π_k)

    subject to θ ≥ 0, Σ_k θ_k = 1.

    Causal Stacking is the special case ν=0, β=0: minimises mean squared error
    of the ensemble prediction against the DR pseudo-outcome.

    Parameters
    ----------
    pseudo_outcomes : array of shape (m,)
        DR/AIPW pseudo-outcomes Γ_i.
    cate_predictions : array of shape (K, m)
        Out-of-fold CATE predictions; one row per component model.
    nu : float
        Interpolation parameter ν ∈ [0, 1]. ν=0 → pure ensemble loss (Causal
        Stacking). ν=1 → weighted average of individual model losses.
        Lan & Syrgkanis (2024) recommend ν = 0.5.
    beta : float
        Temperature for the KL penalty. With the default uniform prior, β has
        no effect because log(1/π_k) = log K is constant on the simplex. β only
        influences the solution when a non-uniform prior is supplied.
    prior : array of shape (K,)
        Prior over models, strictly positive, need not sum to 1 (normalised
        internally). Uniform prior: np.ones(K).
    greedy : bool
        If True, use the greedy approximation (Algorithm 1 from the paper):
        at most 2 nonzero weights via O(K) closed-form 1D line searches.
        If False, use the full SLSQP solver.

    Returns
    -------
    Array of shape (K,) on the probability simplex.
    """
    K, m = cate_predictions.shape

    if K == 1:
        return np.ones(1)

    # Precompute log-prior (normalise prior; only relative values matter)
    log_inv_prior = np.log(prior.sum() / prior)   # log(1/π_k) up to constant

    # Individual model MSEs: shape (K,)
    ind_mse = np.array([np.mean((pseudo_outcomes - cate_predictions[k]) ** 2)
                        for k in range(K)])

    if greedy:
        return _greedy_solver(
            pseudo_outcomes, cate_predictions, nu, beta, log_inv_prior, ind_mse
        )
    return _full_solver(
        pseudo_outcomes, cate_predictions, nu, beta, log_inv_prior, ind_mse
    )


def _greedy_solver(
    Gamma: np.ndarray,
    C: np.ndarray,
    nu: float,
    beta: float,
    log_inv_prior: np.ndarray,
    ind_mse: np.ndarray,
) -> np.ndarray:
    """Greedy approximation: Algorithm 1 of Lan & Syrgkanis (2024).

    Returns a weight vector with at most 2 nonzero entries.
    Runs O(K) closed-form 1D line searches.
    """
    K, m = C.shape

    # Step 1: best single model by individual MSE
    j_star = int(np.argmin(ind_mse))

    # Objective evaluator for a full K-vector weight
    def eval_obj(w: np.ndarray) -> float:
        ensemble = C.T @ w          # shape (m,)
        ens = (1 - nu) * np.mean((Gamma - ensemble) ** 2)
        ind = nu * float(w @ ind_mse)
        kl = (beta / m) * float(w @ log_inv_prior)
        return ens + ind + kl

    w_best = np.zeros(K)
    w_best[j_star] = 1.0
    obj_best = eval_obj(w_best)

    # Step 2: for each other model, find the optimal 2-model mix
    for j in range(K):
        if j == j_star:
            continue

        # 1D line: w(α) = α·e_{j*} + (1-α)·e_j
        # L(α) = C2·α² + C1·α + const,  minimised analytically.
        d = C[j_star] - C[j]              # shape (m,)
        r_j = Gamma - C[j]                # residual of model j

        C2 = (1.0 - nu) * float(np.dot(d, d)) / m
        C1 = (-2.0 * (1.0 - nu) * float(np.dot(r_j, d)) / m
              + nu * (ind_mse[j_star] - ind_mse[j])
              + (beta / m) * (log_inv_prior[j_star] - log_inv_prior[j]))

        if C2 > 0.0:
            alpha = float(np.clip(-C1 / (2.0 * C2), 0.0, 1.0))
        else:
            # ν=1: objective is linear in α; minimum at boundary
            alpha = 1.0 if C1 <= 0.0 else 0.0

        w_cand = np.zeros(K)
        w_cand[j_star] = alpha
        w_cand[j] = 1.0 - alpha

        obj_cand = eval_obj(w_cand)
        if obj_cand < obj_best:
            obj_best = obj_cand
            w_best = w_cand

    return w_best


def _full_solver(
    Gamma: np.ndarray,
    C: np.ndarray,
    nu: float,
    beta: float,
    log_inv_prior: np.ndarray,
    ind_mse: np.ndarray,
) -> np.ndarray:
    """Full trust-region constrained solver on the probability simplex."""
    K, m = C.shape
    CT = C.T  # shape (m, K)

    def objective(w: np.ndarray) -> float:
        ensemble = CT @ w
        ens = (1.0 - nu) * float(np.mean((Gamma - ensemble) ** 2))
        ind = nu * float(w @ ind_mse)
        kl = (beta / m) * float(w @ log_inv_prior)
        return ens + ind + kl

    def gradient(w: np.ndarray) -> np.ndarray:
        ensemble = CT @ w                                       # (m,)
        grad_ens = 2.0 * (1.0 - nu) / m * (C @ (ensemble - Gamma))  # (K,)
        grad_ind = nu * ind_mse                                 # (K,)
        grad_kl = (beta / m) * log_inv_prior                   # (K,)
        return grad_ens + grad_ind + grad_kl

    # Hessian is constant: 2*(1-nu)/m * C @ C^T (linear and KL terms vanish)
    H = 2.0 * (1.0 - nu) / m * (C @ CT)

    def hessian(w: np.ndarray) -> np.ndarray:
        return H

    eq_constraint = LinearConstraint(np.ones((1, K)), 1.0, 1.0)
    bounds = Bounds(lb=np.zeros(K), ub=np.inf)
    w0 = np.ones(K) / K

    result = minimize(
        objective,
        w0,
        jac=gradient,
        hess=hessian,
        method="trust-constr",
        bounds=bounds,
        constraints=eq_constraint,
        options={"gtol": 1e-10, "maxiter": 1000, "verbose": 0},
    )

    # Clip and renormalise to correct small numerical violations
    w = np.maximum(result.x, 0.0)
    w_sum = w.sum()
    if w_sum > 0:
        w /= w_sum
    else:
        w = np.ones(K) / K  # fallback: uniform
    return w


# ---------------------------------------------------------------------------
# QAggregation
# ---------------------------------------------------------------------------


@dataclass
class QAggregation(SupervisedStrategy):
    """Doubly Robust Q-Aggregation (Lan & Syrgkanis, 2024).

    Finds a convex combination of component CATEs by minimising a modified
    DR loss with an optional KL penalty on the simplex.

    The modified loss interpolates between the ensemble loss (ν=0) and the
    weighted average of individual model losses (ν=1):

        L̃_i(θ) = (1-ν)(Γ_i - θ·τ(x_i))² + ν Σ_k θ_k (Γ_i - τ_k(x_i))²

    The full objective adds a cross-entropy regulariser:

        L(θ) = (1/m) Σ_i L̃_i(θ) + (β/m) Σ_k θ_k log(1/π_k)

    **Note on β:** With the default uniform prior, log(1/π_k) = log K is
    constant on the simplex and β has no effect on the optimisation. β only
    changes the solution when a non-uniform prior is supplied. Set β > 0 and
    pass a custom prior to encode domain knowledge about which models are
    likely better.

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
    nu : float
        Interpolation parameter ν ∈ [0, 1]. Lan & Syrgkanis (2024) recommend
        ν = 0.5 (default). ν=0 gives the Causal Stacking objective.
    beta : float
        KL penalty temperature. Default 0 (no penalty). Only effective when
        a non-uniform prior is supplied — see note above.
    prior : array-like of shape (K,) or None
        Prior over the K component models. None (default) → uniform prior.
        Must be strictly positive; normalised internally.
    greedy : bool
        If True, use the greedy approximation (Algorithm 1 in the paper):
        at most 2 nonzero weights via O(K) closed-form 1D line searches.
        Achieves the same theoretical guarantee as the full solver.
        Default False (full SLSQP solver).
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
    >>> from metacausal.aggregation import QAggregation
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
    >>> ens = CausalEnsemble(methods=methods, aggregation=QAggregation())
    >>> _ = ens.fit(X, T, Y, random_state=42)
    >>> ens.ate()
    AteEstimate(ate=..., n_methods=2, aggregation='QAggregation', spread=...)
    """

    nu: float = 0.5
    beta: float = 0.0
    prior: Any = None   # np.ndarray | None
    greedy: bool = False
    pseudo_outcome_fn: Any = field(default=None, repr=False)

    def fit_weights(
        self,
        cate_predictions: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        nuisance: Any,
    ) -> EnsembleWeights:
        """Compute Q-aggregation weights from OOF predictions and nuisance estimates.

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
                method="q_aggregation",
                details={"nu": self.nu, "beta": self.beta, "greedy": self.greedy},
            )
            return self._weights

        # Validate and resolve prior
        if self.prior is None:
            prior = np.ones(K)
        else:
            prior = np.asarray(self.prior, dtype=float)
            if prior.shape != (K,):
                raise ValueError(
                    f"prior must have shape ({K},) to match {K} component models, "
                    f"got shape {prior.shape}"
                )
            if np.any(prior <= 0):
                raise ValueError("prior must be strictly positive for all entries")

        pseudo_outcome_fn = self.pseudo_outcome_fn if self.pseudo_outcome_fn is not None else dr_pseudo_outcome
        pseudo_outcomes = pseudo_outcome_fn(Y, T, nuisance)

        weights = _solve_dr_simplex(
            pseudo_outcomes=pseudo_outcomes,
            cate_predictions=cate_predictions,
            nu=self.nu,
            beta=self.beta,
            prior=prior,
            greedy=self.greedy,
        )

        self._weights = EnsembleWeights(
            weights=weights,
            model_names=[],
            method="q_aggregation",
            details={"nu": self.nu, "beta": self.beta, "greedy": self.greedy},
        )
        return self._weights
