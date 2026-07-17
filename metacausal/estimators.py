"""Data classes for causal estimates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from metacausal._formatting import format_interval, format_scalar, summarize_array
from metacausal.aggregation.weights import EnsembleWeights

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass
class ComponentAteEstimate:
    """Result from a single causal estimator.

    Attributes:
        ate: Average treatment effect point estimate.
        ci_lower: Lower bound of confidence interval (if available).
        ci_upper: Upper bound of confidence interval (if available).
        details: Method-specific additional information.
    """

    ate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    details: dict | None = None

    def __repr__(self) -> str:
        parts = [f"ate={format_scalar(self.ate)}"]
        ci = format_interval(self.ci_lower, self.ci_upper)
        if ci is not None:
            parts.append(f"ci={ci}")
        return f"ComponentAteEstimate({', '.join(parts)})"


# Backward-compatible alias
CausalEstimate = ComponentAteEstimate


@dataclass
class AteEstimate:
    """Result from the ensemble of causal estimators.

    Attributes:
        ate: Aggregated treatment effect point estimate.
        component_estimates: Per-method estimates, keyed by method name.
        aggregation: Aggregation strategy class name.
        component_fit_times: Wall-clock fit time in seconds for each method
            on the full dataset. Does not include bootstrap iterations.
    """

    ate: float
    component_estimates: dict[str, ComponentAteEstimate]
    aggregation: str
    component_fit_times: dict[str, float] | None = field(default=None, repr=False)

    @property
    def component_ates(self) -> dict[str, float]:
        """ATEs from each component method."""
        return {k: v.ate for k, v in self.component_estimates.items()}

    @property
    def spread(self) -> float:
        """Range (max - min) of component ATEs."""
        ates = list(self.component_ates.values())
        return max(ates) - min(ates)

    @property
    def n_methods(self) -> int:
        """Number of methods that produced estimates."""
        return len(self.component_estimates)

    def __repr__(self) -> str:
        return (
            f"AteEstimate(ate={format_scalar(self.ate)}, "
            f"n_methods={self.n_methods}, aggregation={self.aggregation!r}, "
            f"spread={format_scalar(self.spread)})"
        )

    def summary(
        self,
        *,
        digits: int | None = None,
        signed: bool = False,
        show_ci: bool = True,
    ) -> str:
        """Return a formatted, multi-line ATE summary.

        When ``show_ci`` is False, the per-component native confidence
        intervals are omitted, leaving a clean point-estimate table (useful
        when component CIs are unavailable for some methods or when the CI
        story is told separately, e.g. via ``bootstrap()``).
        """
        value_width = max(12, len(format_scalar(self.ate, digits=digits, signed=signed)))
        lines = [
            f"Ensemble ATE ({self.aggregation}): "
            f"{format_scalar(self.ate, digits=digits, signed=signed)}",
            "Components:",
        ]
        for name, est in self.component_estimates.items():
            line = (
                f"  {name:<24} "
                f"{format_scalar(est.ate, digits=digits, signed=signed):>{value_width}}"
            )
            if show_ci:
                ci = format_interval(
                    est.ci_lower,
                    est.ci_upper,
                    digits=digits,
                    signed=signed,
                )
                if ci is not None:
                    line += f"  CI {ci}"
            lines.append(line)
        lines.append(
            "Spread (max - min): "
            f"{format_scalar(self.spread, digits=digits, signed=signed)}"
        )
        return "\n".join(lines)


# Backward-compatible alias
EnsembleEstimate = AteEstimate


@dataclass
class ComponentCateEstimate:
    """CATE result from a single component method.

    Attributes:
        cate: Per-observation treatment effect estimates, shape (n,).
        ci_lower: Lower bounds of confidence/credible intervals, shape (n,).
            From the method's native inference (e.g., EconML analytical CIs,
            stochtree posterior quantiles). None if unavailable.
        ci_upper: Upper bounds, shape (n,). None if unavailable.
        details: Method-specific additional information.
    """

    cate: np.ndarray
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    details: dict | None = None

    def __repr__(self) -> str:
        return f"ComponentCateEstimate({summarize_array(self.cate)})"


@dataclass
class CateEstimate:
    """Ensemble CATE result. Pure data, no model references.

    Attributes:
        cate: Aggregated per-observation treatment effects, shape (n,).
        component_estimates: Per-method CATE results, keyed by method name.
            Only includes methods with ``supports_cate=True``.
        aggregation: Aggregation strategy class name.
        ensemble_weights: Weight vector from the aggregation strategy.
            None for pointwise strategies (Median, Mean).
    """

    cate: np.ndarray
    component_estimates: dict[str, ComponentCateEstimate]
    aggregation: str
    ensemble_weights: EnsembleWeights | None = None

    @property
    def component_cates(self) -> dict[str, np.ndarray]:
        """CATE arrays from each component method."""
        return {k: v.cate for k, v in self.component_estimates.items()}

    @property
    def n_methods(self) -> int:
        """Number of methods that produced CATE estimates."""
        return len(self.component_estimates)

    def __repr__(self) -> str:
        cate = np.asarray(self.cate, dtype=float).ravel()
        return (
            f"CateEstimate(n_obs={cate.size}, "
            f"mean_cate={format_scalar(float(np.mean(cate)))}, "
            f"n_methods={self.n_methods}, aggregation={self.aggregation!r})"
        )

    def summary(
        self,
        *,
        digits: int | None = None,
        signed: bool = False,
    ) -> str:
        """Return a formatted, multi-line CATE summary."""
        del signed  # CATE distribution summaries are unsigned by design.
        lines = [
            f"CATE summary ({self.aggregation})",
            f"Ensemble: {summarize_array(self.cate, digits=digits)}",
            f"Component pool ({self.n_methods}):",
        ]
        for name, est in self.component_estimates.items():
            lines.append(f"  {name:<24} {summarize_array(est.cate, digits=digits)}")
        if self.ensemble_weights is not None:
            lines.append("Weights:")
            width = max(len(name) for name in self.ensemble_weights.model_names)
            for name, weight in zip(
                self.ensemble_weights.model_names,
                self.ensemble_weights.weights,
                strict=True,
            ):
                lines.append(
                    f"  {name:<{width}}  {format_scalar(weight, digits=3)}"
                )
            if self.ensemble_weights.intercept != 0.0:
                lines.append(
                    "Intercept: "
                    f"{format_scalar(self.ensemble_weights.intercept, digits=digits)}"
                )
        return "\n".join(lines)

    def cate_profile(
        self,
        x: np.ndarray,
        *,
        xlabel: str,
        ax: Axes | None = None,
        show_components: bool = True,
        ylim: tuple[float, float] | None = None,
        ensemble_label: str = "Ensemble CATE",
    ) -> Axes:
        """Ensemble CATE along one covariate (no CI band; use
        :meth:`~metacausal.aggregation.weights.BootstrapResult.cate_profile`
        for a bootstrap band).

        Thin wrapper around :func:`metacausal.plots.cate_profile`; see
        there for the full parameter reference. Requires the ``plots``
        extra (``pip install 'metacausal[plots]'``).

        Returns:
            matplotlib Axes the plot was drawn on.
        """
        from metacausal.plots import cate_profile as _cate_profile

        return _cate_profile(
            self, x, xlabel=xlabel, ax=ax, show_components=show_components,
            ylim=ylim, ensemble_label=ensemble_label,
        )
