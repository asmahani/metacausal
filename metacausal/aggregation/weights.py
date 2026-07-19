"""EnsembleWeights and BootstrapResult dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from metacausal._formatting import (
    SummaryStr,
    format_interval,
    format_scalar,
    summarize_array,
    truncate_items,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


def _scaled_percentile_ci(
    dist: np.ndarray,
    point_estimate: float | np.ndarray,
    alpha: float,
    scale: float = 1.0,
    axis: int | None = None,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Politis-Romano (1994) scaled-percentile CI:
    ``point_estimate + scale * percentile(dist - point_estimate, level)``.

    ``scale=1.0`` (the nonparametric case, m == n) makes this algebraically
    reduce to the standard percentile bootstrap. Shared by
    :meth:`CausalEnsemble.bootstrap`'s internal ATE/CATE CI construction
    and :meth:`BootstrapResult.ci_at`, so there is one implementation of
    the formula rather than one per call site.

    Works for both a scalar statistic (``dist`` shape ``(B,)``,
    ``point_estimate`` a float, ``axis=None``) and a per-point statistic
    (``dist`` shape ``(B, n)``, ``point_estimate`` shape ``(n,)``,
    ``axis=0``) -- numpy broadcasts the subtraction along the trailing
    axis in both cases.
    """
    dist = np.asarray(dist, dtype=float)
    centered = dist - point_estimate
    lo = point_estimate + scale * np.percentile(centered, 100 * alpha / 2, axis=axis)
    hi = point_estimate + scale * np.percentile(centered, 100 * (1 - alpha / 2), axis=axis)
    if axis is None:
        return float(lo), float(hi)
    return lo, hi


@dataclass
class EnsembleWeights:
    """Result of weight computation for any non-pointwise aggregation.

    Attributes:
        weights: Per-component weights, shape (K,). Sum to 1 for agreement
            and DR simplex strategies; non-negative for R-Stacking.
        model_names: Adapter names in the same order as weights.
        intercept: Constant CATE shift. Nonzero only for R-Stacking.
        method: Which strategy produced these weights.
        details: Method-specific metadata (e.g., mean_taus for CBA).
    """

    weights: np.ndarray
    model_names: list[str]
    intercept: float = 0.0
    method: str = ""
    details: dict | None = None

    def __repr__(self) -> str:
        pairs = [
            f"{name}={format_scalar(weight, digits=3)}"
            for name, weight in zip(self.model_names, self.weights, strict=True)
        ]
        joined = truncate_items(pairs, max_items=4)
        return (
            f"EnsembleWeights(method={self.method!r}, "
            f"weights=[{joined}], "
            f"intercept={format_scalar(self.intercept, digits=3)})"
        )

    def plot(
        self,
        *,
        ax: Axes | None = None,
        on_uniform: Literal["warn", "error", "ignore"] = "warn",
        sort: bool = True,
    ) -> Axes:
        """Bar chart of these aggregation weights.

        Thin wrapper around :func:`metacausal.plots.weights`; see there
        for the full parameter reference. Requires the ``plots`` extra
        (``pip install 'metacausal[plots]'``).

        Returns:
            matplotlib Axes the plot was drawn on.
        """
        from metacausal.plots import weights as _weights

        return _weights(self, ax=ax, on_uniform=on_uniform, sort=sort)


@dataclass
class BootstrapResult:
    """Bootstrap inference results for both ATE and CATE.

    Point estimates (ate, cate) come from the original fit(), not from
    averaging bootstrap replicates — consistent with standard bootstrap
    CI practice.

    Attributes:
        ate: Point estimate from the original fit.
        ate_ci_lower: Lower bound of bootstrap CI for ATE.
        ate_ci_upper: Upper bound of bootstrap CI for ATE.
        boot_ates: Bootstrap distribution of ensemble ATEs, shape (B,).
        cate: Point estimate from the original fit, shape (n,).
            None if no CATE-capable adapters are available.
        cate_ci_lower: Pointwise lower CI bounds, shape (n,). None if
            no CATE-capable adapters.
        cate_ci_upper: Pointwise upper CI bounds, shape (n,). None if
            no CATE-capable adapters.
        boot_cates: Bootstrap CATE distributions, shape (B, n). None if
            no CATE-capable adapters.
        component_boot_ates: Per-component ATE distributions,
            {adapter_name: shape (B,)}. Bootstrap samples, not point
            estimates — contrast with ``component_ate_estimates``.
        component_ate_estimates: Full-sample per-component ATE point
            estimates, {adapter_name: ComponentAteEstimate}. Carries the
            point ATE plus any native CI the method provides (e.g.,
            EconML analytical CIs). Computed during the original fit();
            distinct from the bootstrap distributions in
            ``component_boot_ates``.
        component_cate_estimates: Full-sample per-component CATE point
            estimates, {adapter_name: ComponentCateEstimate}. None if
            no CATE-capable adapter is available. Each entry carries
            the point CATE array and any native per-grid-point CI.
        n_boot: Number of bootstrap replicates requested.
        n_failed: Number of replicates that failed entirely.
        alpha: Significance level used for CIs.
        aggregation: Strategy class name that produced these results.
        ensemble_weights: Weights from the original fit (None for
            pointwise strategies).
        method: Bootstrap resampling scheme — ``"nonparametric"`` (n-out-of-n
            with replacement, the standard Efron bootstrap) or
            ``"subsample"`` (m-out-of-n without replacement, T-stratified;
            CIs use the Politis–Romano scaled-percentile correction).
        subsample_m: Subsample size used when ``method="subsample"``;
            ``None`` for nonparametric.
        n_train: Original training sample size. Recorded so ``scale``
            and :meth:`ci_at` can reconstruct the Politis-Romano scale
            factor after the fact, without needing ``n`` passed back in.
    """

    # ATE
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    boot_ates: np.ndarray  # shape (B,)

    # CATE (optional — only populated when CATE-capable adapters exist)
    cate: np.ndarray | None
    cate_ci_lower: np.ndarray | None
    cate_ci_upper: np.ndarray | None
    boot_cates: np.ndarray | None = field(repr=False, default=None)  # shape (B, n)

    # Per-component ATE distributions
    component_boot_ates: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    # Full-sample per-component point estimates (with native per-method CIs).
    # Distinct from the bootstrap distributions above: these come from the
    # original fit() on the full training data, not from replicate resamples.
    component_ate_estimates: dict[str, "ComponentAteEstimate"] = field(
        repr=False, default_factory=dict
    )
    component_cate_estimates: dict[str, "ComponentCateEstimate"] | None = field(
        repr=False, default=None
    )

    # Metadata
    n_boot: int = 0
    n_failed: int = 0
    alpha: float = 0.05
    aggregation: str = ""
    ensemble_weights: EnsembleWeights | None = None
    method: str = "nonparametric"
    subsample_m: int | None = None
    n_train: int = 0

    @property
    def scale(self) -> float:
        """Politis-Romano scale factor: ``sqrt(subsample_m / n_train)`` for
        ``method="subsample"``, ``1.0`` otherwise (including when
        ``n_train`` wasn't recorded, e.g. a hand-built ``BootstrapResult``)."""
        if self.method == "subsample" and self.subsample_m is not None and self.n_train > 0:
            return float(np.sqrt(self.subsample_m / self.n_train))
        return 1.0

    def ci_at(
        self,
        level: float | None = None,
        *,
        dist: np.ndarray | None = None,
        point_estimate: float | None = None,
    ) -> tuple[float, float]:
        """Confidence interval at an arbitrary level, reusing this bootstrap
        run's already-computed replicate distribution(s) -- no re-fit needed.

        By default, reproduces the ensemble's own ATE CI (at ``level``
        instead of the level fixed when ``bootstrap()`` was called), using
        the same Politis-Romano scaled-percentile formula (with the
        ``method="subsample"`` scale correction applied automatically via
        :attr:`scale`).

        Pass ``dist`` to get a CI for any other 1-d bootstrap distribution
        carried by this result -- e.g. ``component_boot_ates["BCF"]`` for a
        per-component CI -- or one you've derived yourself (e.g. a
        per-replicate mean across components, for an aggregation strategy
        the ensemble was never actually fit with). ``dist`` and
        ``point_estimate`` must be overridden together: a bootstrap
        distribution can only be centered correctly on the point estimate
        of the *same* statistic it resamples, and there is no way to infer
        the right point estimate for an arbitrary distribution.

        Parameters:
            level: Coverage level, e.g. ``0.90`` for a 90% CI. Defaults to
                ``1 - self.alpha`` (the level this result was already
                constructed at).
            dist: Bootstrap replicate distribution, shape ``(B,)``.
                Defaults to ``self.boot_ates``.
            point_estimate: Center of the interval. Defaults to
                ``self.ate``. Required (and only meaningful) together
                with ``dist``.

        Returns:
            ``(ci_lower, ci_upper)`` tuple.

        Raises:
            ValueError: If exactly one of ``dist``/``point_estimate`` is
                given.
        """
        if (dist is None) != (point_estimate is None):
            raise ValueError(
                "dist and point_estimate must be overridden together -- "
                "a bootstrap distribution can only be centered correctly "
                "on the point estimate of the same statistic it resamples."
            )
        level = (1.0 - self.alpha) if level is None else level
        alpha = 1.0 - level
        dist = self.boot_ates if dist is None else dist
        theta_hat = self.ate if point_estimate is None else float(point_estimate)
        return _scaled_percentile_ci(dist, theta_hat, alpha, self.scale)

    def __repr__(self) -> str:
        level = round(100 * (1 - self.alpha))
        ci = format_interval(self.ate_ci_lower, self.ate_ci_upper)
        return (
            f"BootstrapResult(ate={format_scalar(self.ate)}, "
            f"ci={ci}, level={level}%, n_boot={self.n_boot}, "
            f"n_failed={self.n_failed}, method={self.method!r})"
        )

    def component_ate_summary(self) -> pd.DataFrame:
        """Tabular summary of per-component bootstrap ATE statistics.

        Returns a DataFrame with one row per component adapter, in the
        insertion order of ``component_boot_ates``. Each row reports the
        bootstrap-mean ATE and the ``1 - alpha`` CI, via :meth:`ci_at`
        (matching the CI convention used by the ensemble itself, including
        the ``method="subsample"`` scale correction).

        Returns
        -------
        pandas.DataFrame
            Columns:

            * ``name``: adapter name.
            * ``mean``: bootstrap-mean ATE.
            * ``lo``: lower CI bound.
            * ``hi``: upper CI bound.

        Raises
        ------
        ValueError
            If ``component_boot_ates`` is empty (bootstrap not run or
            no component distributions recorded).
        """
        if not self.component_boot_ates:
            raise ValueError(
                "component_boot_ates is empty; run CausalEnsemble.bootstrap "
                "before calling component_ate_summary()."
            )
        rows = []
        for name, boot in self.component_boot_ates.items():
            point = self.component_ate_estimates.get(name)
            point_estimate = point.ate if point is not None else float(np.mean(boot))
            lo, hi = self.ci_at(dist=boot, point_estimate=point_estimate)
            rows.append({
                "name": name,
                "mean": float(np.mean(boot)),
                "lo": lo,
                "hi": hi,
            })
        return pd.DataFrame(rows, columns=["name", "mean", "lo", "hi"])

    def summary(
        self,
        *,
        digits: int | None = None,
        signed: bool = False,
    ) -> str:
        """Return a formatted, multi-line bootstrap summary."""
        level = round(100 * (1 - self.alpha))
        summary = self.component_ate_summary()
        name_width = max(4, max(len(name) for name in summary["name"]))
        value_width = max(
            12,
            max(
                len(format_scalar(v, digits=digits, signed=signed))
                for v in (
                    summary["mean"].tolist()
                    + summary["lo"].tolist()
                    + summary["hi"].tolist()
                    + [self.ate, self.ate_ci_lower, self.ate_ci_upper]
                )
            ),
        )
        lines = [
            (
                f"Bootstrap ATE summary ({self.method}, {level}% CI, "
                f"n_boot={self.n_boot}, n_failed={self.n_failed})"
            ),
            f"{'name':<{name_width}} "
            f"{'mean':>{value_width}} "
            f"{'lo':>{value_width}} "
            f"{'hi':>{value_width}}",
        ]
        for row in summary.itertuples(index=False):
            lines.append(
                f"{row.name:<{name_width}} "
                f"{format_scalar(row.mean, digits=digits, signed=signed):>{value_width}} "
                f"{format_scalar(row.lo, digits=digits, signed=signed):>{value_width}} "
                f"{format_scalar(row.hi, digits=digits, signed=signed):>{value_width}}"
            )
        lines.extend(
            [
                "",
                (
                    f"Ensemble {level}% CI: "
                    f"{format_interval(self.ate_ci_lower, self.ate_ci_upper, digits=digits, signed=signed)}"
                ),
                (
                    f"Point estimate: "
                    f"{format_scalar(self.ate, digits=digits, signed=signed)}"
                ),
            ]
        )
        if self.cate is not None:
            lines.append(f"CATE: {summarize_array(self.cate, digits=digits)}")
        return SummaryStr("\n".join(lines))

    def forest(
        self,
        *,
        ax: Axes | None = None,
        order: Literal["value", "alpha", "ci_width", "input"] = "value",
        ensemble_label: str = "Ensemble",
        show_zero: bool = True,
    ) -> Axes:
        """Forest plot of per-component and ensemble ATEs.

        Thin wrapper around :func:`metacausal.plots.forest`; see there for
        the full parameter reference. Requires the ``plots`` extra
        (``pip install 'metacausal[plots]'``).

        Returns:
            matplotlib Axes the plot was drawn on.
        """
        from metacausal.plots import forest as _forest

        return _forest(
            self, ax=ax, order=order, ensemble_label=ensemble_label,
            show_zero=show_zero,
        )

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
        """Ensemble CATE along one covariate, with bootstrap CI band.

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
