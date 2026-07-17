"""Bar chart of aggregation weights."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from metacausal.aggregation import EnsembleWeights
from metacausal.plots._base import ENSEMBLE_COLOR

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from metacausal import CausalEnsemble


_UNIFORM_TOL = 1e-8


def weights(
    source: CausalEnsemble | EnsembleWeights,
    *,
    ax: Axes | None = None,
    on_uniform: Literal["warn", "error", "ignore"] = "warn",
    sort: bool = True,
) -> Axes:
    """Bar chart of aggregation weights.

    Works naturally on supervised aggregators (causal stacking, BMA,
    DR simplex, R-stacking, CBA) that produce informative weights.
    Pointwise aggregators (``Mean``, ``Median``, ``TrimmedMean``) either
    expose no weights at all, or would produce uniform bars; see
    ``on_uniform``.

    Parameters
    ----------
    source
        Either a fitted ``CausalEnsemble`` — the weights are read from
        ``source.aggregation.ensemble_weights`` — or a precomputed
        ``EnsembleWeights`` instance.
    ax
        Existing axes to draw on. If ``None``, a new figure is created.
    on_uniform
        Behaviour when weights are effectively uniform (max pairwise
        deviation below a small tolerance):

        * ``"warn"`` (default): emit a ``UserWarning`` and render
          uniform bars.
        * ``"error"``: raise ``ValueError``.
        * ``"ignore"``: render uniform bars silently.
    sort
        If ``True``, sort bars by descending weight. If ``False``,
        preserve the order in ``EnsembleWeights.model_names``.

    Returns
    -------
    Axes
        The axes the plot was drawn on.

    Raises
    ------
    ValueError
        If ``source`` is a ``CausalEnsemble`` whose aggregation has no
        ``ensemble_weights`` (pure pointwise strategies), or if
        ``on_uniform="error"`` and the weights are uniform.

    Examples
    --------
    Works directly on a standalone ``EnsembleWeights`` (as produced by any
    supervised or agreement strategy's ``ensemble_weights`` after fitting),
    with no ``CausalEnsemble`` required for the plot itself:

    >>> from metacausal.aggregation import EnsembleWeights
    >>> from metacausal.plots import weights
    >>> ew = EnsembleWeights(weights=[0.7, 0.3], model_names=["a", "b"], method="cba")
    >>> ax = weights(ew)
    >>> ax.get_xlabel()
    'Weight'

    .. plot::
        :include-source: False

        from metacausal.aggregation import EnsembleWeights
        from metacausal.plots import weights

        ew = EnsembleWeights(weights=[0.7, 0.3], model_names=["a", "b"], method="cba")
        weights(ew)
    """
    if isinstance(source, EnsembleWeights):
        ew = source
    else:
        ew = source.aggregation.ensemble_weights
        if ew is None:
            raise ValueError(
                f"Aggregation {type(source.aggregation).__name__!r} is a "
                "pointwise strategy and has no ensemble_weights. Use a "
                "supervised aggregator (e.g. CausalStacking) to produce "
                "a weights plot."
            )

    w = np.asarray(ew.weights, dtype=float)
    names = list(ew.model_names)
    if w.size == 0:
        raise ValueError("EnsembleWeights has no weights to plot.")

    if float(w.max() - w.min()) < _UNIFORM_TOL:
        if on_uniform == "error":
            raise ValueError(
                f"Weights are effectively uniform "
                f"(all ≈ {float(w.mean()):.4g}). A bar chart will not be "
                "informative. Use a supervised aggregator, or pass "
                "on_uniform='warn' / 'ignore' to render anyway."
            )
        if on_uniform == "warn":
            warnings.warn(
                f"Weights are effectively uniform (all ≈ {float(w.mean()):.4g}); "
                "the bar chart will show equal-length bars.",
                UserWarning,
                stacklevel=2,
            )

    if sort:
        order_idx = np.argsort(-w)
        names = [names[i] for i in order_idx]
        w = w[order_idx]

    k = len(w)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, max(2.2, 0.35 * k + 1.0)))

    active = w > _UNIFORM_TOL
    colors = [ENSEMBLE_COLOR if a else "lightgray" for a in active]

    # Reverse so largest bar is at the top when matplotlib draws bottom-up.
    ax.barh(names[::-1], w[::-1], color=colors[::-1])
    uniform = 1.0 / k
    ax.axvline(
        uniform,
        color="black",
        linestyle=":",
        linewidth=0.7,
        label=f"uniform = 1/{k}",
    )
    ax.set_xlabel("Weight")
    ax.set_xlim(0, max(float(w.max()) * 1.15, uniform * 1.1))
    ax.legend(loc="lower right")
    method = ew.method or type(source).__name__
    ax.set_title(f"Ensemble weights — {method}")

    return ax
