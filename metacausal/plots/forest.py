"""Forest plot of component and ensemble ATEs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from metacausal.plots._base import (
    COMPONENT_COLOR,
    ENSEMBLE_COLOR,
    ZERO_REFERENCE_COLOR,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from metacausal.aggregation import BootstrapResult


def forest(
    boot_result: BootstrapResult,
    *,
    ax: Axes | None = None,
    order: Literal["value", "alpha", "ci_width", "input"] = "value",
    ensemble_label: str = "Ensemble",
    show_zero: bool = True,
) -> Axes:
    """Forest plot of per-component and ensemble ATEs.

    Each component row shows the bootstrap mean ATE and its
    ``1 - alpha`` CI (lower/upper bounds from percentiles of
    ``component_boot_ates[name]``). The ensemble row uses
    ``boot_result.ate`` and ``boot_result.ate_ci_{lower,upper}`` and is
    always drawn at the top of the plot.

    Parameters
    ----------
    boot_result
        Output of ``CausalEnsemble.bootstrap(...)``. Must have a
        non-empty ``component_boot_ates`` mapping.
    ax
        Existing axes to draw on. If ``None``, a new figure is created.
    order
        Row ordering for component rows (top-down):

        * ``"value"`` (default): ascending bootstrap-mean ATE.
        * ``"alpha"``: alphabetical by component name.
        * ``"ci_width"``: ascending CI width.
        * ``"input"``: insertion order of ``component_boot_ates``.
    ensemble_label
        Y-tick label for the ensemble row.
    show_zero
        If ``True``, draw a dotted vertical reference line at ATE = 0.

    Returns
    -------
    Axes
        The axes the plot was drawn on.

    Raises
    ------
    ValueError
        If ``boot_result.component_boot_ates`` is empty.

    Examples
    --------
    Works directly on a standalone ``BootstrapResult`` (as returned by
    ``CausalEnsemble.bootstrap(...)``), constructed by hand here with toy
    bootstrap distributions for two components:

    >>> import numpy as np
    >>> from metacausal.aggregation import BootstrapResult
    >>> from metacausal.plots import forest
    >>> boot = BootstrapResult(
    ...     ate=5.0, ate_ci_lower=1.0, ate_ci_upper=9.0,
    ...     boot_ates=np.array([3.0, 5.0, 7.0]),
    ...     cate=None, cate_ci_lower=None, cate_ci_upper=None,
    ...     component_boot_ates={
    ...         "a": np.array([2.0, 4.0, 6.0]),
    ...         "b": np.array([4.0, 6.0, 8.0]),
    ...     },
    ... )
    >>> ax = forest(boot)
    >>> ax.get_xlabel()
    'ATE'

    .. plot::
        :include-source: False

        import numpy as np
        from metacausal.aggregation import BootstrapResult
        from metacausal.plots import forest

        boot = BootstrapResult(
            ate=5.0, ate_ci_lower=1.0, ate_ci_upper=9.0,
            boot_ates=np.array([3.0, 5.0, 7.0]),
            cate=None, cate_ci_lower=None, cate_ci_upper=None,
            component_boot_ates={
                "a": np.array([2.0, 4.0, 6.0]),
                "b": np.array([4.0, 6.0, 8.0]),
            },
        )
        forest(boot)
    """
    summary = boot_result.component_ate_summary()

    if order == "value":
        summary = summary.sort_values("mean", ascending=True, kind="stable")
    elif order == "alpha":
        summary = summary.sort_values("name", ascending=True, kind="stable")
    elif order == "ci_width":
        summary = summary.assign(_w=summary["hi"] - summary["lo"])
        summary = summary.sort_values("_w", ascending=True, kind="stable")
        summary = summary.drop(columns="_w")
    elif order == "input":
        pass
    else:  # pragma: no cover — Literal guards at type level
        raise ValueError(f"Unknown order: {order!r}")

    names = summary["name"].tolist()
    means = summary["mean"].to_numpy()
    lo = summary["lo"].to_numpy()
    hi = summary["hi"].to_numpy()

    n_c = len(names)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, max(2.5, 0.35 * (n_c + 1) + 1.0)))

    comp_ys = np.arange(n_c)
    ensemble_y = n_c  # drawn at the top (highest y)

    ax.hlines(comp_ys, lo, hi, color=COMPONENT_COLOR, linewidth=2)
    ax.plot(means, comp_ys, "o", color=COMPONENT_COLOR, markersize=5)

    ax.hlines(
        [ensemble_y],
        [boot_result.ate_ci_lower],
        [boot_result.ate_ci_upper],
        color=ENSEMBLE_COLOR,
        linewidth=3,
    )
    ax.plot([boot_result.ate], [ensemble_y], "D",
            color=ENSEMBLE_COLOR, markersize=8)

    if show_zero:
        ax.axvline(0, color=ZERO_REFERENCE_COLOR, linestyle=":", linewidth=0.6)

    ax.set_yticks(list(comp_ys) + [ensemble_y])
    ax.set_yticklabels(names + [ensemble_label])
    ax.get_yticklabels()[-1].set_color(ENSEMBLE_COLOR)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("ATE")

    return ax
