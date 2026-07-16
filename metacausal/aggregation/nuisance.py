"""Nuisance model fitting and pseudo-outcome utilities for supervised aggregation.

All nuisance estimates are out-of-fold (OOF): no observation's prediction is
made by a model that was trained on that observation.

For CrossFitSplit (Q folds), every observation gets an OOF prediction and
NuisanceEstimates arrays are fully populated with shape (n,).

For TrainAvgSplit (1 fold), only the averaging-set positions (test_indices[0])
get predictions; the training-set positions are left as np.nan. The calling
code (_fit_supervised in ensemble.py) is responsible for subsetting to the
averaging-set observations before passing data to fit_weights.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone

from metacausal.aggregation.splitting import FoldSpec


# ---------------------------------------------------------------------------
# NuisanceEstimates
# ---------------------------------------------------------------------------


@dataclass
class NuisanceEstimates:
    """Out-of-fold nuisance model predictions.

    Attributes
    ----------
    e_hat : array of shape (n,)
        Propensity scores P(T=1|X), trimmed to [propensity_trim, 1-propensity_trim].
        Already trimmed at fit time — do not untrim before computing pseudo-outcomes.
    mu1_hat : array of shape (n,)
        Out-of-fold predictions of E[Y|X, T=1].
    mu0_hat : array of shape (n,)
        Out-of-fold predictions of E[Y|X, T=0].

    Notes
    -----
    For TrainAvgSplit, positions corresponding to the training set are np.nan.
    Only the averaging-set positions (fold_spec.test_indices[0]) are populated.
    """

    e_hat: np.ndarray
    mu1_hat: np.ndarray
    mu0_hat: np.ndarray

    @property
    def m_hat(self) -> np.ndarray:
        """Marginal conditional mean E[Y|X] = e*mu1 + (1-e)*mu0."""
        return self.e_hat * self.mu1_hat + (1 - self.e_hat) * self.mu0_hat


# ---------------------------------------------------------------------------
# fit_nuisance
# ---------------------------------------------------------------------------


def fit_nuisance(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    fold_spec: FoldSpec,
    propensity_model=None,
    outcome_model=None,
    propensity_trim: float = 0.01,
    random_state=None,
    outcome_type: str | None = None,
) -> NuisanceEstimates:
    """Fit nuisance models via cross-fitting and return out-of-fold predictions.

    For each fold j in fold_spec:
      1. Clone and fit propensity model on the training indices.
      2. Clone and fit outcome model on treated training units -> mu1 on test indices.
      3. Clone and fit outcome model on control training units -> mu0 on test indices.
      4. Clip propensity to [propensity_trim, 1 - propensity_trim].

    For CrossFitSplit, all n positions are filled. For TrainAvgSplit, only
    fold_spec.test_indices[0] positions are filled; the rest remain np.nan.

    Parameters
    ----------
    X : array of shape (n, p)
    T : array of shape (n,)
        Binary treatment assignment (0 or 1).
    Y : array of shape (n,)
        Observed outcome.
    fold_spec : FoldSpec
        Output of CrossFitSplit.split() or TrainAvgSplit.split().
    propensity_model : sklearn classifier or None
        Model for P(T=1|X). Must support predict_proba. Default: HistGradientBoostingClassifier.
    outcome_model : sklearn regressor / classifier or None
        Model for E[Y|X, T]. Cloned separately for treated and control fits.
        For continuous Y, must be a regressor; ``mu_hat`` is read from
        ``predict()``. For binary Y, must be a classifier with
        ``predict_proba``; ``mu_hat`` is read from ``predict_proba(X)[:, 1]``.
        ``None`` selects an outcome-type-appropriate default
        (HistGradientBoostingRegressor or HistGradientBoostingClassifier).
    propensity_trim : float
        Clip propensity scores to [trim, 1-trim] to enforce overlap.
    outcome_type : "continuous", "binary", or None
        How to interpret Y. ``None`` (default) auto-detects from the value
        set of Y via :func:`metacausal.infer_outcome_type`.

    Returns
    -------
    NuisanceEstimates with out-of-fold predictions. e_hat is already trimmed.
    """
    from metacausal.defaults import default_outcome_model, default_propensity_model
    from metacausal.outcome_type import infer_outcome_type

    X = np.asarray(X)
    T = np.asarray(T)
    Y = np.asarray(Y)
    n = len(T)

    if outcome_type is None:
        outcome_type = infer_outcome_type(Y)
    elif outcome_type not in ("continuous", "binary"):
        raise ValueError(
            f"outcome_type must be 'continuous', 'binary', or None; "
            f"got {outcome_type!r}."
        )

    e_hat = np.full(n, np.nan)
    mu1_hat = np.full(n, np.nan)
    mu0_hat = np.full(n, np.nan)

    base_prop = propensity_model if propensity_model is not None else default_propensity_model()
    base_out = (
        outcome_model
        if outcome_model is not None
        else default_outcome_model(outcome_type)
    )

    # Validate user-supplied outcome_model matches the outcome type.
    out_has_proba = hasattr(base_out, "predict_proba")
    if outcome_type == "binary" and not out_has_proba:
        raise ValueError(
            "Binary outcome detected but the supplied outcome_model "
            f"{type(base_out).__name__!r} does not implement predict_proba. "
            "Pass a classifier (e.g. HistGradientBoostingClassifier)."
        )
    if outcome_type == "continuous" and out_has_proba:
        raise ValueError(
            "Continuous outcome detected but the supplied outcome_model "
            f"{type(base_out).__name__!r} is a classifier (has predict_proba). "
            "Pass a regressor."
        )

    # 3 model slots per fold: [prop, mu1, mu0]
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31, size=(fold_spec.n_folds, 3)).tolist()

    def _predict_outcome(model, X_test):
        if outcome_type == "binary":
            return model.predict_proba(X_test)[:, 1]
        return model.predict(X_test)

    for j in range(fold_spec.n_folds):
        train_idx = fold_spec.train_indices[j]
        test_idx = fold_spec.test_indices[j]

        X_train, X_test = X[train_idx], X[test_idx]
        T_train = T[train_idx]
        Y_train = Y[train_idx]

        treated_mask = T_train == 1
        control_mask = T_train == 0

        if treated_mask.sum() == 0:
            raise ValueError(
                f"Fold {j} training set contains no treated units. "
                "Reduce n_folds or use TrainAvgSplit."
            )
        if control_mask.sum() == 0:
            raise ValueError(
                f"Fold {j} training set contains no control units. "
                "Reduce n_folds or use TrainAvgSplit."
            )

        prop_seed, mu1_seed, mu0_seed = seeds[j]

        # --- propensity ---
        prop_model = clone(base_prop)
        if hasattr(prop_model, "random_state"):
            prop_model.random_state = int(prop_seed)
        prop_model.fit(X_train, T_train)
        raw_e = prop_model.predict_proba(X_test)[:, 1]
        e_hat[test_idx] = np.clip(raw_e, propensity_trim, 1.0 - propensity_trim)

        # --- treated outcome model (mu1) ---
        mu1_model = clone(base_out)
        if hasattr(mu1_model, "random_state"):
            mu1_model.random_state = int(mu1_seed)
        mu1_model.fit(X_train[treated_mask], Y_train[treated_mask])
        mu1_hat[test_idx] = _predict_outcome(mu1_model, X_test)

        # --- control outcome model (mu0) ---
        mu0_model = clone(base_out)
        if hasattr(mu0_model, "random_state"):
            mu0_model.random_state = int(mu0_seed)
        mu0_model.fit(X_train[control_mask], Y_train[control_mask])
        mu0_hat[test_idx] = _predict_outcome(mu0_model, X_test)

    return NuisanceEstimates(e_hat=e_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat)


# ---------------------------------------------------------------------------
# Pseudo-outcome utilities
# ---------------------------------------------------------------------------


def dr_pseudo_outcome(
    Y: np.ndarray,
    T: np.ndarray,
    nuisance: NuisanceEstimates,
) -> np.ndarray:
    """Compute the DR/AIPW pseudo-outcome for each observation.

    ::

        Gamma_i = (mu1_hat - mu0_hat)
                  + T * (Y - mu1_hat) / e_hat
                  - (1 - T) * (Y - mu0_hat) / (1 - e_hat)

    This is doubly robust: unbiased for tau*(X) whenever either the propensity
    model or the outcome models are correctly specified. With oracle nuisance,
    E[Gamma_i | X_i] = tau*(X_i).

    Parameters
    ----------
    Y : array of shape (n,)
    T : array of shape (n,)
        Binary treatment assignment.
    nuisance : NuisanceEstimates
        Must have e_hat already trimmed (no division by zero).

    Returns
    -------
    Array of shape (n,).
    """
    Y = np.asarray(Y)
    T = np.asarray(T)

    imputation = nuisance.mu1_hat - nuisance.mu0_hat
    treated_correction = T * (Y - nuisance.mu1_hat) / nuisance.e_hat
    control_correction = (1 - T) * (Y - nuisance.mu0_hat) / (1 - nuisance.e_hat)

    return imputation + treated_correction - control_correction


def robinson_residuals(
    Y: np.ndarray,
    T: np.ndarray,
    nuisance: NuisanceEstimates,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Robinson decomposition residuals.

    Y_tilde = Y - m_hat(X)
    W_tilde = T - e_hat(X)

    Used by R-Stacking to reformulate the CATE estimation as a weighted
    least squares problem (Nie & Wager, 2021).

    Parameters
    ----------
    Y : array of shape (n,)
    T : array of shape (n,)
    nuisance : NuisanceEstimates

    Returns
    -------
    (Y_tilde, W_tilde), each of shape (n,).
    """
    Y = np.asarray(Y)
    T = np.asarray(T)

    Y_tilde = Y - nuisance.m_hat
    W_tilde = T - nuisance.e_hat

    return Y_tilde, W_tilde
