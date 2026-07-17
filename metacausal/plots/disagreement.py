"""Heatmap of pairwise component CATE disagreement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from metacausal import CausalEnsemble


def disagreement(
    ensemble: CausalEnsemble,
    X: np.ndarray,
    *,
    ax: Axes | None = None,
    metric: Literal["spearman", "pearson", "rmse"] = "spearman",
    cluster: bool = False,
    annotate: bool = True,
) -> Axes:
    """Pairwise disagreement between component CATEs evaluated on ``X``.

    Computes each CATE-capable component's CATE on ``X``, forms a
    ``(K, K)`` matrix of pairwise agreement under ``metric``, and
    renders it as a heatmap with optional cell annotations.

    Parameters
    ----------
    ensemble
        A fitted ``CausalEnsemble`` with at least two CATE-capable
        components.
    X
        Covariates to evaluate each component's CATE on,
        shape ``(n, p)``. Typically the training data or a held-out
        sample.
    ax
        Existing axes to draw on. If ``None``, a new figure is created.
    metric
        Pairwise metric:

        * ``"spearman"`` (default): rank correlation of unit-level
          CATEs. Robust to scale differences between components.
        * ``"pearson"``: linear correlation of unit-level CATEs.
        * ``"rmse"``: root mean squared difference. Scale-aware and
          dominated by components with extreme predictions.
    cluster
        If ``True``, reorder rows/columns by hierarchical clustering.
        Correlation metrics use ``1 - |corr|`` as the distance; RMSE
        is used directly. Requires ``scipy.cluster.hierarchy``.
    annotate
        If ``True``, write each cell's value inside the heatmap.

    Returns
    -------
    Axes
        The axes the plot was drawn on.

    Raises
    ------
    ValueError
        If ``ensemble`` has fewer than two CATE-capable components.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    >>> from metacausal import CausalEnsemble
    >>> from metacausal.adapters import GenericCATEAdapter
    >>> from metacausal.datasets import load_lalonde
    >>> from metacausal.plots import disagreement
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
    >>> ens = CausalEnsemble(methods=methods)
    >>> _ = ens.fit(X, T, Y, random_state=42)
    >>> ax = disagreement(ens, X)
    >>> ax.get_title()
    'Component CATE spearman agreement'

    .. plot::
        :include-source: False

        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
        from metacausal import CausalEnsemble
        from metacausal.adapters import GenericCATEAdapter
        from metacausal.datasets import load_lalonde
        from metacausal.plots import disagreement

        X, T, Y = load_lalonde()

        def fit_linear(X, T, Y, **kwargs):
            treated = T == 1
            m1 = LinearRegression().fit(X[treated], Y[treated])
            m0 = LinearRegression().fit(X[~treated], Y[~treated])
            return (m1, m0)

        def fit_hgb(X, T, Y, **kwargs):
            treated = T == 1
            m1 = HGBR(max_iter=20).fit(X[treated], Y[treated])
            m0 = HGBR(max_iter=20).fit(X[~treated], Y[~treated])
            return (m1, m0)

        def cate_fn(state, X):
            m1, m0 = state
            return m1.predict(X) - m0.predict(X)

        methods = [
            GenericCATEAdapter(fit_linear, cate_fn, name="linear"),
            GenericCATEAdapter(fit_hgb, cate_fn, name="hgb"),
        ]
        ens = CausalEnsemble(methods=methods)
        ens.fit(X, T, Y, random_state=42)
        disagreement(ens, X)
    """
    cate_est = ensemble.cate(X)
    component_cates = cate_est.component_cates
    names = list(component_cates.keys())
    if len(names) < 2:
        raise ValueError(
            f"disagreement() needs at least 2 CATE-capable components, "
            f"got {len(names)}."
        )
    # Shape (n, K) — each column is one component's CATE over the units.
    M = np.column_stack([component_cates[name] for name in names])

    if metric == "pearson":
        matrix = np.corrcoef(M, rowvar=False)
        label = "Pearson r"
        cmap = "RdBu_r"
        vmin, vmax = -1.0, 1.0
    elif metric == "spearman":
        ranks = np.apply_along_axis(rankdata, 0, M)
        matrix = np.corrcoef(ranks, rowvar=False)
        label = "Spearman ρ"
        cmap = "RdBu_r"
        vmin, vmax = -1.0, 1.0
    elif metric == "rmse":
        diff = M[:, :, None] - M[:, None, :]  # (n, K, K)
        matrix = np.sqrt(np.mean(diff ** 2, axis=0))
        label = "RMSE"
        cmap = "viridis_r"
        vmin, vmax = 0.0, float(matrix.max()) if matrix.max() > 0 else 1.0
    else:  # pragma: no cover — Literal guards at type level
        raise ValueError(f"Unknown metric: {metric!r}")

    if cluster:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        if metric == "rmse":
            distance = matrix.copy()
        else:
            distance = 1.0 - np.abs(matrix)
        np.fill_diagonal(distance, 0.0)
        # Symmetrise against floating point drift.
        distance = (distance + distance.T) / 2.0
        order = leaves_list(linkage(squareform(distance, checks=False),
                                    method="average"))
        matrix = matrix[np.ix_(order, order)]
        names = [names[i] for i in order]

    k = len(names)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.5, 0.55 * k + 2.5),
                                      max(4.0, 0.55 * k + 2.0)))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    if annotate:
        mid = (vmin + vmax) / 2.0 if metric == "rmse" else 0.0
        span = max(abs(vmax - mid), abs(vmin - mid), 1e-12)
        for i in range(k):
            for j in range(k):
                val = matrix[i, j]
                # Contrast: white text on darker cells.
                rel = abs(val - mid) / span
                color = "white" if rel > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    ax.set_title(f"Component CATE {metric} agreement")

    return ax
