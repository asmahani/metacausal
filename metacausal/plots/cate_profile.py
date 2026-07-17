"""CATE profile along one covariate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from metacausal.aggregation import BootstrapResult
from metacausal.estimators import CateEstimate
from metacausal.plots._base import (
    COMPONENT_COLOR,
    ENSEMBLE_COLOR,
    ZERO_REFERENCE_COLOR,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def cate_profile(
    source: BootstrapResult | CateEstimate,
    x: np.ndarray,
    *,
    xlabel: str,
    ax: Axes | None = None,
    show_components: bool = True,
    ylim: tuple[float, float] | None = None,
    ensemble_label: str = "Ensemble CATE",
) -> Axes:
    """Ensemble CATE along one covariate, with optional bootstrap CI.

    The x-axis is the varied covariate; the y-axis is CATE. Individual
    component CATEs are optionally overlaid in grey. A shaded
    ``1 - alpha`` pointwise CI is drawn only when a ``BootstrapResult``
    is supplied.

    Parameters
    ----------
    source
        Either:

        * A ``BootstrapResult`` produced by
          ``CausalEnsemble.bootstrap(X_grid, ...)`` — draws the CI band
          plus ensemble and component lines. Must have a non-``None``
          ``cate``.
        * A ``CateEstimate`` produced by
          ``CausalEnsemble.cate(X_grid)`` — draws ensemble and
          component lines only, no band.

        The grid length must match ``len(x)``.
    x
        Values of the varied covariate, shape ``(grid_size,)``. These
        become the x-axis coordinates.
    xlabel
        Axis label for ``x`` (e.g., ``"re74 (1974 earnings)"``).
    ax
        Existing axes to draw on. If ``None``, a new figure is created.
    show_components
        If ``True``, overlay per-component CATEs as thin grey lines.
        Only drawn when ``source`` carries component estimates.
    ylim
        Optional y-axis limits. Useful when a single component has
        extreme CATE values that compress the ensemble signal.
    ensemble_label
        Legend label for the ensemble line.

    Returns
    -------
    Axes
        The axes the plot was drawn on.

    Raises
    ------
    ValueError
        If ``source`` carries no CATE data (e.g., a ``BootstrapResult``
        whose ``cate`` is ``None``), or if ``len(x)`` does not match
        the grid length on ``source``.

    Examples
    --------
    Works directly on a standalone ``CateEstimate`` (as returned by
    ``CausalEnsemble.cate(X_grid)``), constructed by hand here with a toy
    grid and a single component:

    >>> import numpy as np
    >>> from metacausal.estimators import CateEstimate, ComponentCateEstimate
    >>> from metacausal.plots import cate_profile
    >>> x = np.linspace(0, 1, 5)
    >>> result = CateEstimate(
    ...     cate=x * 2,
    ...     component_estimates={"a": ComponentCateEstimate(cate=x * 2 + 0.1)},
    ...     aggregation="Median",
    ... )
    >>> ax = cate_profile(result, x, xlabel="x")
    >>> ax.get_ylabel()
    'CATE'

    .. plot::
        :include-source: False

        import numpy as np
        from metacausal.estimators import CateEstimate, ComponentCateEstimate
        from metacausal.plots import cate_profile

        x = np.linspace(0, 1, 5)
        result = CateEstimate(
            cate=x * 2,
            component_estimates={"a": ComponentCateEstimate(cate=x * 2 + 0.1)},
            aggregation="Median",
        )
        cate_profile(result, x, xlabel="x")
    """
    x = np.asarray(x)

    if isinstance(source, BootstrapResult):
        if source.cate is None:
            raise ValueError(
                "BootstrapResult has no CATE data. Bootstrap with an X_grid "
                "argument, or pass a CateEstimate from ensemble.cate(X_grid)."
            )
        ensemble_cate = np.asarray(source.cate)
        ci_lower = (
            np.asarray(source.cate_ci_lower)
            if source.cate_ci_lower is not None else None
        )
        ci_upper = (
            np.asarray(source.cate_ci_upper)
            if source.cate_ci_upper is not None else None
        )
        if source.component_cate_estimates is not None:
            component_cates: dict[str, np.ndarray] = {
                name: np.asarray(est.cate)
                for name, est in source.component_cate_estimates.items()
            }
        else:
            component_cates = {}
    elif isinstance(source, CateEstimate):
        ensemble_cate = np.asarray(source.cate)
        ci_lower = None
        ci_upper = None
        component_cates = {
            name: np.asarray(est.cate)
            for name, est in source.component_estimates.items()
        }
    else:  # pragma: no cover
        raise TypeError(
            f"source must be BootstrapResult or CateEstimate, "
            f"got {type(source).__name__}"
        )

    if ensemble_cate.shape[0] != x.shape[0]:
        raise ValueError(
            f"Grid length mismatch: source has {ensemble_cate.shape[0]} "
            f"points but x has {x.shape[0]}."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.0))

    has_ci = ci_lower is not None and ci_upper is not None
    if has_ci:
        ax.fill_between(
            x, ci_lower, ci_upper,
            color=ENSEMBLE_COLOR, alpha=0.22,
            label="pointwise CI",
        )

    ax.plot(x, ensemble_cate, color=ENSEMBLE_COLOR, linewidth=2.2,
            label=ensemble_label)

    if show_components and component_cates:
        for i, (_name, cate_i) in enumerate(component_cates.items()):
            if cate_i.shape[0] != x.shape[0]:
                raise ValueError(
                    "Component CATE length mismatch for "
                    f"{_name}: {cate_i.shape[0]} vs {x.shape[0]}."
                )
            label = "components" if i == 0 else None
            ax.plot(x, cate_i, color=COMPONENT_COLOR, linewidth=0.7,
                    alpha=0.55, label=label)

    ax.axhline(0, color=ZERO_REFERENCE_COLOR, linestyle=":", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CATE")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")

    return ax
