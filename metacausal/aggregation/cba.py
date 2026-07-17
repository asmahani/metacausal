"""Consensus Based Averaging (CBA) aggregation strategy."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from metacausal.aggregation.base import AgreementStrategy
from metacausal.aggregation.weights import EnsembleWeights


@dataclass
class CBA(AgreementStrategy):
    """Consensus Based Averaging (Machluf et al., 2024).

    Selects a high-agreement subset of component models via Kendall's tau
    rank correlation and averages them with uniform weights.

    The algorithm:

    1. Compute pairwise Kendall's tau between all component CATE predictions.
    2. Compute mean tau per model (how well each model agrees with others).
    3. Sort models descending by mean tau.
    4. Find the "knee" — the largest drop in consecutive mean taus.
    5. Select models above the knee and average with uniform weights.

    Complexity is O(K² n log n) for pairwise Kendall's tau; practical for
    K < ~20 component models.

    **Source:** Machluf et al. (2024). "Robust CATE Estimation Using Novel
    Ensemble Methods." arXiv:2407.03690, Appendix A.

    Parameters
    ----------
    eps : float
        Tolerance for the flat-differences guard. If
        ``max(diff) - min(diff) < eps``, all consecutive drops are considered
        equal and all models are selected (no meaningful knee exists).
        Metacausal-specific default; not from the original paper.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    >>> from metacausal import CausalEnsemble
    >>> from metacausal.adapters import GenericCATEAdapter
    >>> from metacausal.aggregation import CBA
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
    >>> ens = CausalEnsemble(methods=methods, aggregation=CBA())
    >>> _ = ens.fit(X, T, Y, random_state=42)
    >>> ens.ate()
    AteEstimate(ate=..., n_methods=2, aggregation='CBA', spread=...)
    """

    eps: float = 0.01

    def compute_weights(
        self,
        cate_matrix: np.ndarray,
        model_names: list[str],
    ) -> EnsembleWeights:
        """Compute CBA weights from training-data CATE predictions.

        Called once during ``CausalEnsemble.fit()``.

        Parameters
        ----------
        cate_matrix : array of shape (K, n)
            Training-data CATE predictions, one row per model.
        model_names : list[str]
            Adapter names in the same order as rows of cate_matrix.

        Returns
        -------
        EnsembleWeights with uniform weights over the selected subset.
        """
        from scipy.stats import kendalltau

        K, n = cate_matrix.shape

        if K == 1:
            self._weights = EnsembleWeights(
                weights=np.array([1.0]),
                model_names=list(model_names),
                method="cba",
                details={"mean_taus": {model_names[0]: 1.0}, "n_selected": 1, "selected_models": [model_names[0]]},
            )
            return self._weights

        # Step 1: pairwise Kendall's tau
        tau_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1, K):
                tau, _ = kendalltau(cate_matrix[i], cate_matrix[j])
                if np.isnan(tau):
                    raise ValueError(
                        f"Kendall's tau is NaN for models '{model_names[i]}' and "
                        f"'{model_names[j]}'. This usually means one or both "
                        f"produced constant CATE predictions."
                    )
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau

        # Step 2: mean tau per model (diagonal is 0, divide by K-1)
        mean_taus = tau_matrix.sum(axis=1) / (K - 1)

        # Step 3: sort descending
        sorted_indices = np.argsort(-mean_taus)
        sorted_mean_taus = mean_taus[sorted_indices]

        if K == 2:
            # Edge case: both models have identical mean tau; use both
            selected = sorted_indices.tolist()
        else:
            # Step 4: knee detection — argmin of consecutive differences
            diffs = np.diff(sorted_mean_taus)  # all <= 0 since sorted descending
            if diffs.max() - diffs.min() < self.eps:
                # No meaningful knee; use all models
                selected = sorted_indices.tolist()
            else:
                m = int(np.argmin(diffs)) + 1  # models at indices 0..m-1 are selected
                selected = sorted_indices[:m].tolist()

        if len(selected) == 1:
            warnings.warn(
                f"CBA selected only 1 model ('{model_names[selected[0]]}'). "
                f"The ensemble degenerates to a single model.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Build uniform weights over selected models; zero for excluded
        weights = np.zeros(K)
        for idx in selected:
            weights[idx] = 1.0 / len(selected)

        mean_taus_dict = {model_names[i]: float(mean_taus[i]) for i in range(K)}

        self._weights = EnsembleWeights(
            weights=weights,
            model_names=list(model_names),
            method="cba",
            details={
                "mean_taus": mean_taus_dict,
                "n_selected": len(selected),
                "selected_models": [model_names[i] for i in selected],
            },
        )
        return self._weights
