"""CausalEnsemble — the main class."""

from __future__ import annotations

import copy
import time
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from metacausal._formatting import truncate_items
from metacausal._parallel import parallel_map
from metacausal._warnings import (
    BootstrapWarning,
    ComponentExclusionWarning,
    ComponentFailureWarning,
)
from metacausal.adapters.base import CausalEstimator
from metacausal.adapters.econml import EconMLAdapter, _is_econml
from metacausal.adapters.causalml import CausalMLAdapter, _is_causalml
from metacausal.adapters.generic import GenericATEAdapter, GenericCATEAdapter
from metacausal.aggregation import (
    _STRING_FACTORIES,
    AggregationStrategy,
    AgreementStrategy,
    PointwiseStrategy,
    SupervisedStrategy,
)
from metacausal.aggregation.weights import BootstrapResult, EnsembleWeights
from metacausal.estimators import (
    AteEstimate,
    CateEstimate,
    ComponentAteEstimate,
    ComponentCateEstimate,
)


def _fit_one_adapter(
    m: CausalEstimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    seed: int | None,
    kwargs: dict,
) -> tuple[str, CausalEstimator | None, float, str | None]:
    """Fit one adapter; return (name, adapter_or_None, fit_time, exc_repr_or_None).

    Module-level (not a method) so it is picklable for loky workers.
    Warnings are not emitted here — the caller re-emits based on the
    returned exception repr so they surface in the parent process.
    """
    t0 = time.perf_counter()
    try:
        adapter = copy.deepcopy(m)
        adapter.fit(X, T, Y, random_state=seed, **kwargs)
        return m.name, adapter, time.perf_counter() - t0, None
    except Exception as e:
        return m.name, None, time.perf_counter() - t0, repr(e)


def _fit_one_fold_method(
    j: int,
    m: CausalEstimator,
    X_train: np.ndarray,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    test_idx: np.ndarray,
    seed: int | None,
    kwargs: dict,
) -> tuple[int, str, np.ndarray | None, np.ndarray | None, str | None]:
    """Fit ``m`` on fold ``j``'s train split and predict CATE on the test split.

    Returns (fold_index, name, test_idx, cate_on_test_or_None, exc_repr_or_None).
    Module-level for loky pickling. Callers use the fold index + test_idx to
    scatter the returned CATE slice back into the full-length OOF vector.
    """
    try:
        adapter = copy.deepcopy(m)
        adapter.fit(X_train, T_train, Y_train, random_state=seed, **kwargs)
        cate_slice = adapter.cate(X_test).cate
        return j, m.name, test_idx, cate_slice, None
    except Exception as e:
        return j, m.name, test_idx, None, repr(e)


def _resolve_subsample_size(n: int, subsample_size: float | int) -> int:
    """Resolve ``subsample_size`` into an integer m with 1 <= m < n.

    Float in (0, 1) is interpreted as a fraction of n; int is interpreted
    as m directly. m must be strictly less than n — at m = n no
    subsampling is possible without duplicates, defeating the purpose of
    the subsample method. ``replace=False`` would also collapse to a
    deterministic permutation.
    """
    if isinstance(subsample_size, bool):
        raise TypeError("subsample_size must be float in (0, 1) or int, not bool")
    if isinstance(subsample_size, (int, np.integer)):
        m = int(subsample_size)
    elif isinstance(subsample_size, (float, np.floating)):
        if not (0.0 < subsample_size < 1.0):
            raise ValueError(
                f"subsample_size as a float must be in (0, 1); got {subsample_size}"
            )
        m = max(1, int(round(float(subsample_size) * n)))
    else:
        raise TypeError(
            f"subsample_size must be float in (0, 1) or int; "
            f"got {type(subsample_size).__name__}"
        )
    if m < 1 or m >= n:
        raise ValueError(
            f"subsample_size resolved to m={m}; must satisfy 1 <= m < n (n={n})"
        )
    return m


def _stratified_subsample(
    rng: np.random.Generator, T: np.ndarray, m: int
) -> np.ndarray:
    """Draw a size-m index array stratified by T, no replacement.

    Each treatment level contributes m * n_level / n units (rounded);
    the largest stratum absorbs any rounding remainder so the total is
    exactly m. If a stratum's quota exceeds its size it is capped, and
    the deficit is topped up uniformly from the remaining pool.

    No replacement within or across strata: the returned indices are
    pairwise distinct, which preserves cross-fit honesty in any
    downstream component or supervised wrapper.
    """
    n = len(T)
    levels, counts = np.unique(T, return_counts=True)
    # Largest stratum absorbs the rounding remainder; iterate over the
    # rest first.
    order = np.argsort(-counts)
    last_pos = order[0]
    head = order[1:]

    indices: list[np.ndarray] = []
    used = 0
    for pos in head:
        level = levels[pos]
        count = counts[pos]
        m_level = int(round(m * count / n))
        m_level = min(m_level, count)
        level_idx = np.where(T == level)[0]
        chosen = rng.choice(level_idx, size=m_level, replace=False)
        indices.append(chosen)
        used += m_level

    last_level = levels[last_pos]
    last_idx = np.where(T == last_level)[0]
    m_last = max(0, min(m - used, len(last_idx)))
    chosen_last = rng.choice(last_idx, size=m_last, replace=False)
    indices.append(chosen_last)

    out = np.concatenate(indices) if indices else np.array([], dtype=int)
    if len(out) < m:
        # Cap-induced shortfall: top up uniformly from unused pool.
        used_mask = np.zeros(n, dtype=bool)
        used_mask[out] = True
        remaining = np.where(~used_mask)[0]
        deficit = min(m - len(out), len(remaining))
        if deficit > 0:
            top_up = rng.choice(remaining, size=deficit, replace=False)
            out = np.concatenate([out, top_up])
    rng.shuffle(out)
    return out


class CausalEnsemble:
    """Ensemble of causal estimators with robust aggregation.

    Orchestrates multiple causal ML methods and aggregates their treatment
    effect estimates. The median is the default aggregation rule, providing
    a 50% breakdown point against catastrophic failure of individual methods.

    Parameters:
        methods: List of causal estimators. Accepts:

            - Objects implementing :class:`CausalEstimator` protocol
            - EconML estimator instances (auto-detected and wrapped)
            - :class:`DoubleMLAdapter` instances (explicit)
            - :class:`GenericATEAdapter` or :class:`GenericCATEAdapter` instances
            - Lists/tuples of 2 or 3 callables, auto-wrapped as
              :class:`GenericCATEAdapter`

            If ``None``, uses :func:`default_methods` — and which pool
            it returns depends on the outcome type detected at fit time.
        aggregation: Aggregation strategy. Either a string alias
            (``"median"``, ``"mean"``, ``"trimmed_mean"``, ``"cba"``) or a
            strategy object implementing
            :class:`~metacausal.aggregation.AggregationStrategy`.
        outcome_type: How to interpret Y. ``"auto"`` (default) detects
            from the value set of Y at fit time
            (:func:`metacausal.infer_outcome_type`): Y ⊆ {0, 1} or boolean
            is binary, otherwise continuous. ``"continuous"`` and
            ``"binary"`` force the corresponding interpretation.
            ``"binary"`` with non-binary Y raises; ``"continuous"`` with
            binary Y is silently accepted (a "give me the linear-
            probability ATE" request).

    Outcome-type lifecycle: at fit time the ensemble resolves the outcome
    type from Y, materialises the right default pool if no methods were
    supplied, drops any component whose ``supported_outcome_types`` does
    not include the detected type (with one summary warning naming the
    drops), and validates each survivor's nuisance configuration. For
    binary outcomes the estimand is the **risk difference**.

    Examples:
        Fit two components and compute the ensemble ATE with a
        full-pipeline bootstrap. ``estimate()`` (not shown) offers a
        one-call fit + ate/bootstrap convenience wrapper.

        >>> from metacausal import CausalEnsemble
        >>> from metacausal.datasets import load_lalonde
        >>> from metacausal.adapters import DoubleMLAdapter
        >>> from econml.dml import CausalForestDML
        >>> from doubleml import DoubleMLIRM
        >>> from sklearn.ensemble import (
        ...     HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        ... )
        >>> X, T, Y = load_lalonde()
        >>> ens = CausalEnsemble([
        ...     CausalForestDML(
        ...         model_y=HistGradientBoostingRegressor(max_iter=50),
        ...         model_t=HistGradientBoostingClassifier(max_iter=50),
        ...         discrete_treatment=True,
        ...         n_estimators=50,
        ...         n_jobs=1,
        ...     ),
        ...     DoubleMLAdapter(
        ...         DoubleMLIRM,
        ...         ml_g=HistGradientBoostingRegressor(max_iter=50),
        ...         ml_m=HistGradientBoostingClassifier(max_iter=50),
        ...     ),
        ... ])
        >>> _ = ens.fit(X, T, Y, random_state=42)
        >>> result = ens.ate()
        >>> boot = ens.bootstrap(n_boot=5, random_state=42)  # small n_boot for a fast example
    """

    def __init__(
        self,
        methods: list | None = None,
        aggregation: str | AggregationStrategy = "median",
        outcome_type: str = "auto",
    ) -> None:
        if isinstance(aggregation, str):
            if aggregation not in _STRING_FACTORIES:
                raise ValueError(
                    f"aggregation must be one of {list(_STRING_FACTORIES)} "
                    f"or a strategy object, got {aggregation!r}"
                )
            aggregation = _STRING_FACTORIES[aggregation]()

        self.aggregation = aggregation

        if outcome_type not in ("auto", "continuous", "binary"):
            raise ValueError(
                f"outcome_type must be 'auto', 'continuous', or 'binary'; "
                f"got {outcome_type!r}."
            )
        self._outcome_type_request = outcome_type

        if methods is None:
            # Defer default-pool materialization to fit() so we can build
            # the right pool from the detected outcome type. Pre-fit
            # introspection of method_names returns an empty list in this
            # state — call default_methods() directly to inspect a pool.
            self._user_supplied_methods = False
            self._initial_wrapped: list[CausalEstimator] | None = None
        else:
            self._user_supplied_methods = True
            self._initial_wrapped = self._wrap_and_validate(methods)

        # _wrapped_methods is the active pool used by fit() and the
        # introspection properties. Set up-front for user-supplied
        # methods; populated at fit() time for default-pool callers.
        self._wrapped_methods: list[CausalEstimator] = (
            list(self._initial_wrapped) if self._initial_wrapped else []
        )

        # Outcome type resolved at fit time; None until then.
        self._outcome_type: str | None = None

        # Fitted state (populated by fit())
        self._fitted_adapters: list[CausalEstimator] | None = None
        self._X_train: np.ndarray | None = None
        self._T_train: np.ndarray | None = None
        self._Y_train: np.ndarray | None = None
        self._component_fit_times: dict[str, float] | None = None
        self._fit_random_state: int | None = None
        self._fit_kwargs: dict[str, Any] = {}
        self._is_fitted = False

        # Cached training CATE predictions (used by AgreementStrategy and bootstrap)
        self._cached_cate_model_names: list[str] | None = None
        self._cached_train_cate_matrix: np.ndarray | None = None

        # Bootstrap method selector (set by bootstrap() per call).
        self._boot_method: str = "nonparametric"
        self._boot_m: int | None = None
        self._boot_stratify: bool = True

        # Cached OOF artifacts from supervised fit (used by strategy override)
        self._cached_oof_cate_predictions: np.ndarray | None = None
        self._cached_oof_cate_model_names: list[str] | None = None
        self._cached_nuisance: Any | None = None  # NuisanceEstimates
        # Exact (X, T, Y) arrays passed to fit_weights — shape matches
        # _cached_oof_cate_predictions columns (n for CrossFitSplit, avg_n for TrainAvgSplit)
        self._cached_fit_weights_X: np.ndarray | None = None
        self._cached_fit_weights_T: np.ndarray | None = None
        self._cached_fit_weights_Y: np.ndarray | None = None

    def __repr__(self) -> str:
        aggregation = type(self.aggregation).__name__
        outcome_type = self._outcome_type or self._outcome_type_request
        if self._wrapped_methods:
            component_summary = (
                f"n_components={len(self._wrapped_methods)}, "
                f"components=[{truncate_items(self.method_names)}]"
            )
        elif self._user_supplied_methods:
            component_summary = "n_components=0, components=[]"
        else:
            component_summary = "components='default pool deferred to fit()'"
        return (
            f"CausalEnsemble(aggregation={aggregation!r}, "
            f"outcome_type={outcome_type!r}, fitted={self._is_fitted}, "
            f"{component_summary})"
        )

    def summary(self) -> str:
        """Return a formatted, multi-line summary of ensemble state."""
        aggregation = type(self.aggregation).__name__
        outcome_type = self._outcome_type or self._outcome_type_request
        lines = [
            "CausalEnsemble",
            f"Aggregation: {aggregation}",
            f"Outcome type: {outcome_type}",
            f"Fitted: {self._is_fitted}",
        ]
        if self._wrapped_methods:
            lines.append(f"Components ({len(self._wrapped_methods)}):")
            for name in self.method_names:
                lines.append(f"  {name}")
            lines.append(
                f"CATE-capable components: {len(self.cate_method_names)}"
            )
        elif self._user_supplied_methods:
            lines.append("Components (0):")
        else:
            lines.append("Components: default pool deferred to fit()")
        return "\n".join(lines)

    def _wrap(self, m: Any) -> CausalEstimator:
        """Auto-detect and wrap a method if needed."""
        if isinstance(m, CausalEstimator):
            return m
        if _is_causalml(m):
            return CausalMLAdapter(m)
        if _is_econml(m):
            return EconMLAdapter(m)
        if (
            isinstance(m, (list, tuple))
            and len(m) in (2, 3)
            and all(callable(fn) for fn in m)
        ):
            fn_fit, fn_cate = m[:2]
            fn_ate = m[2] if len(m) == 3 else None
            return GenericCATEAdapter(fn_fit, fn_cate, fn_ate=fn_ate)
        if callable(m):
            name = getattr(m, "__name__", "callable")
            return GenericATEAdapter(m, name=name)

        raise TypeError(
            f"Cannot wrap {type(m).__name__} as a causal estimator. "
            f"Pass an EconML estimator, a DoubleMLAdapter, a callable, "
            f"a 2- or 3-callable list/tuple, or an object implementing "
            f"the CausalEstimator protocol."
        )

    def _wrap_and_validate(self, methods: list) -> list[CausalEstimator]:
        """Wrap a list of methods and check that names are unique."""
        wrapped: list[CausalEstimator] = []
        seen: set[str] = set()
        for m in methods:
            w = self._wrap(m)
            if w.name in seen:
                raise ValueError(
                    f"Duplicate adapter name {w.name!r}. Each component must "
                    f"have a unique name. Pass a name= kwarg to the adapter "
                    f"constructor to disambiguate (e.g., "
                    f"EconMLAdapter(model, name='MyModel_2'))."
                )
            seen.add(w.name)
            wrapped.append(w)
        return wrapped

    def _resolve_outcome_type(self, Y: np.ndarray) -> str:
        """Determine the outcome type for this fit, honoring an override."""
        from metacausal.outcome_type import infer_outcome_type

        detected = infer_outcome_type(Y)
        request = self._outcome_type_request
        if request == "auto":
            return detected
        if request == "binary" and detected != "binary":
            raise ValueError(
                f"outcome_type='binary' was requested but Y is not binary "
                f"(detected {detected!r}). Either pass outcome_type='auto' "
                f"or fix the supplied Y."
            )
        # outcome_type='continuous' with binary Y is accepted silently —
        # a legitimate "give me the linear-probability ATE" request.
        return request

    def _filter_and_validate(self, detected: str) -> list[CausalEstimator]:
        """Filter the active pool by capability, then per-component validate.

        Components whose ``supported_outcome_types`` lacks ``detected`` are
        dropped (one warning summarising drops). Survivors run their own
        ``validate_outcome_type(detected)``, which may raise on a
        misconfigured nuisance learner.
        """
        surviving: list[CausalEstimator] = []
        dropped: list[tuple[str, tuple[str, ...]]] = []
        for m in self._wrapped_methods:
            supported = tuple(getattr(m, "supported_outcome_types", ("continuous",)))
            if detected in supported:
                surviving.append(m)
            else:
                dropped.append((m.name, supported))

        if dropped:
            details = ", ".join(f"{n} (supports {s})" for n, s in dropped)
            warnings.warn(
                f"Outcome type {detected!r} is not supported by "
                f"{len(dropped)} component(s); dropping: {details}.",
                ComponentExclusionWarning,
                stacklevel=3,
            )

        if not surviving:
            raise RuntimeError(
                f"No component supports the detected outcome type "
                f"{detected!r}. All {len(dropped)} components were "
                f"filtered out."
            )

        for m in surviving:
            m.validate_outcome_type(detected)

        return surviving

    @property
    def method_names(self) -> list[str]:
        """Names of all component methods."""
        return [m.name for m in self._wrapped_methods]

    @property
    def cate_method_names(self) -> list[str]:
        """Names of component methods that support CATE."""
        return [m.name for m in self._wrapped_methods if m.supports_cate]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        random_state: int | None = None,
        n_jobs: int = 1,
        **kwargs,
    ) -> CausalEnsemble:
        """Fit all component methods on the given data.

        Parameters:
            X: Covariate matrix, shape (n, p).
            T: Treatment vector, shape (n,).
            Y: Outcome vector, shape (n,).
            random_state: Seed for reproducibility. Controls component
                estimator randomness.
            n_jobs: Parallel workers for component fits. For a
                :class:`SupervisedStrategy`, this also parallelizes the
                Q×K cross-fitting loop. Inner levels always run
                sequentially; if called inside an already-parallel
                context, this value is ignored.
            **kwargs: Forwarded to each adapter's ``fit()`` call.

        Returns:
            self (for chaining: ``ens.fit(X, T, Y).ate()``).
        """
        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)

        # Resolve outcome type from Y (or honor a non-auto override).
        # Default-pool callers materialize the right pool; user-supplied
        # methods are reset to their initial wrap so a refit on a
        # different Y type re-runs filtering against the new type.
        detected = self._resolve_outcome_type(Y)
        self._outcome_type = detected
        if not self._user_supplied_methods:
            from metacausal.defaults import default_methods

            self._wrapped_methods = self._wrap_and_validate(
                default_methods(detected)
            )
        else:
            assert self._initial_wrapped is not None
            self._wrapped_methods = list(self._initial_wrapped)
        self._wrapped_methods = self._filter_and_validate(detected)

        if isinstance(self.aggregation, SupervisedStrategy):
            self._fit_supervised(X, T, Y, random_state, n_jobs=n_jobs, **kwargs)
        else:
            self._fit_simple(X, T, Y, random_state, n_jobs=n_jobs, **kwargs)

        self._X_train = X
        self._T_train = T
        self._Y_train = Y
        self._fit_random_state = random_state
        self._fit_kwargs = kwargs
        self._is_fitted = True

        # Cache training CATE predictions unconditionally. Needed for:
        # - AgreementStrategy.compute_weights (post-fit, below)
        # - Strategy override in cate()/ate() (Step 5b)
        cate_adapters = [a for a in self._fitted_adapters if a.supports_cate]
        if cate_adapters:
            self._cached_cate_model_names = [a.name for a in cate_adapters]
            self._cached_train_cate_matrix = np.stack(
                [a.cate(X).cate for a in cate_adapters], axis=0
            )
        else:
            self._cached_cate_model_names = []
            self._cached_train_cate_matrix = None

        # Post-fit: compute agreement-based weights if applicable
        if isinstance(self.aggregation, AgreementStrategy):
            if self._cached_train_cate_matrix is None:
                raise RuntimeError(
                    f"Agreement-based aggregation strategies require at least one "
                    f"CATE-capable component method, but none are available. "
                    f"Strategy: {type(self.aggregation).__name__}."
                )
            self.aggregation.compute_weights(
                self._cached_train_cate_matrix,
                self._cached_cate_model_names,
            )

        return self

    def _run_fit_loop(
        self,
        methods: list[CausalEstimator],
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        random_state: int | None,
        n_jobs: int = 1,
        **kwargs,
    ) -> tuple[list[CausalEstimator], dict[str, float]]:
        """Fit a list of adapters and return (fitted_adapters, component_fit_times).

        Adapters that raise are skipped with a RuntimeWarning. The caller is
        responsible for checking whether the returned list is non-empty.

        ``n_jobs`` controls parallel dispatch via :func:`parallel_map`.
        Results are returned in the original ``methods`` order regardless of
        completion order, so the seed-per-adapter mapping is deterministic.
        """
        n = len(methods)
        if random_state is not None:
            rng = np.random.default_rng(random_state)
            seeds: list[int | None] = rng.integers(0, 2**32 - 1, size=n).tolist()
        else:
            seeds = [None] * n

        tasks = [(m, X, T, Y, s, kwargs) for m, s in zip(methods, seeds)]
        results = parallel_map(n_jobs, _fit_one_adapter, tasks)

        fitted: list[CausalEstimator] = []
        times: dict[str, float] = {}
        for name, adapter, elapsed, exc in results:
            if adapter is None:
                warnings.warn(
                    f"Method '{name}' failed: {exc}",
                    ComponentFailureWarning,
                    stacklevel=2,
                )
                continue
            fitted.append(adapter)
            times[name] = elapsed

        return fitted, times

    def _fit_simple(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        random_state: int | None,
        n_jobs: int = 1,
        **kwargs,
    ) -> None:
        """Fit all wrapped methods on the full dataset."""
        fitted, times = self._run_fit_loop(
            self._wrapped_methods, X, T, Y, random_state, n_jobs=n_jobs, **kwargs
        )
        if not fitted:
            raise RuntimeError("All component methods failed.")
        self._fitted_adapters = fitted
        self._component_fit_times = times

    def _fit_supervised(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        random_state: int | None,
        n_jobs: int = 1,
        **kwargs,
    ) -> None:
        """Supervised fit path: cross-fitting, nuisance, weight optimization, retrain.

        Only CATE-capable adapters participate. ATE-only adapters are excluded
        with a warning and do not appear in _fitted_adapters after this call.
        """
        ate_only = [m for m in self._wrapped_methods if not m.supports_cate]
        if ate_only:
            warnings.warn(
                "Supervised aggregation uses only CATE-capable adapters. "
                f"The following ATE-only adapters will be skipped: "
                f"{[m.name for m in ate_only]}.",
                ComponentExclusionWarning,
                stacklevel=3,
            )

        if not any(m.supports_cate for m in self._wrapped_methods):
            raise RuntimeError(
                "Supervised aggregation requires at least one CATE-capable adapter."
            )

        result = self._run_supervised_pipeline(
            X, T, Y, random_state, self.aggregation, n_jobs=n_jobs, **kwargs
        )
        if result is None:
            raise RuntimeError(
                "Supervised fit pipeline failed. All CATE-capable adapters or the "
                "nuisance fitting failed — check warnings above for details."
            )

        self._fitted_adapters = result["fitted_adapters"]
        self._component_fit_times = result["component_fit_times"]
        self._cached_oof_cate_predictions = result["oof_cate_predictions"]
        self._cached_oof_cate_model_names = result["oof_cate_model_names"]
        self._cached_nuisance = result["nuisance"]
        self._cached_fit_weights_X = result["fit_weights_X"]
        self._cached_fit_weights_T = result["fit_weights_T"]
        self._cached_fit_weights_Y = result["fit_weights_Y"]

    def _run_supervised_pipeline(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        random_state: int | None,
        strategy: SupervisedStrategy,
        n_jobs: int = 1,
        **kwargs,
    ) -> dict | None:
        """Core supervised pipeline: cross-fitting, nuisance, weights, retrain.

        Shared by _fit_supervised() and the supervised bootstrap replicate.
        Operates on arbitrary (X, T, Y) without modifying self.

        Parameters
        ----------
        strategy : SupervisedStrategy
            fit_weights() is called on this object, populating its _weights.
            For bootstrap replicates, pass a deep-copied instance per call to
            avoid weight contamination across replicates.

        Returns
        -------
        Dict with keys: fitted_adapters, component_fit_times,
        oof_cate_predictions, oof_cate_model_names, nuisance,
        fit_weights_X, fit_weights_T, fit_weights_Y.
        Returns None on any unrecoverable failure (warnings already emitted).
        """
        from metacausal.aggregation.nuisance import NuisanceEstimates, fit_nuisance

        cate_methods = [m for m in self._wrapped_methods if m.supports_cate]
        if not cate_methods:
            return None

        # Fold assignments
        fold_spec = strategy.split.split(T, random_state=random_state)
        n = len(T)
        K = len(cate_methods)

        # Cross-fitting: OOF CATE predictions
        oof: dict[str, np.ndarray] = {
            m.name: np.full(n, np.nan) for m in cate_methods
        }

        # Fold-level seeds from a sub-stream independent of retrain seeds
        if random_state is not None:
            xfit_seed = int(
                np.random.SeedSequence(random_state).spawn(1)[0].generate_state(1)[0]
            )
            xfit_rng = np.random.default_rng(xfit_seed)
            fold_seeds = xfit_rng.integers(
                0, 2**32 - 1, size=(fold_spec.n_folds, K)
            ).tolist()
        else:
            fold_seeds = [[None] * K for _ in range(fold_spec.n_folds)]

        # Flatten Q×K into a single task list for parallel dispatch.
        tasks = []
        for j in range(fold_spec.n_folds):
            train_idx = fold_spec.train_indices[j]
            test_idx = fold_spec.test_indices[j]
            for k, m in enumerate(cate_methods):
                tasks.append((
                    j, m,
                    X[train_idx], T[train_idx], Y[train_idx],
                    X[test_idx], test_idx,
                    fold_seeds[j][k], kwargs,
                ))
        results = parallel_map(n_jobs, _fit_one_fold_method, tasks)

        # Under Q3 = "drop-entirely": any fold-level failure removes the
        # method from all folds, even folds that succeeded. Collect first,
        # then scatter only surviving methods' slices into `oof`.
        failed_reasons: dict[str, tuple[int, str]] = {}
        for j_r, name, _test_idx, _slice, exc in results:
            if exc is not None and name not in failed_reasons:
                failed_reasons[name] = (j_r, exc)

        for name, (j_fail, exc) in failed_reasons.items():
            warnings.warn(
                f"Method '{name}' failed in fold {j_fail} during cross-fitting: {exc}. "
                "Dropping from all folds.",
                ComponentFailureWarning,
                stacklevel=3,
            )
            oof.pop(name, None)

        for _j_r, name, test_idx_r, cate_slice, exc in results:
            if exc is not None or name in failed_reasons:
                continue
            oof[name][test_idx_r] = cate_slice

        if not oof:
            return None

        # Nuisance fitting (injectable: strategy.fit_nuisance_fn overrides default)
        nuisance_fn = strategy.fit_nuisance_fn if strategy.fit_nuisance_fn is not None else fit_nuisance
        nuisance_seed = (
            int(np.random.SeedSequence(random_state).spawn(2)[1].generate_state(1)[0])
            if random_state is not None else None
        )
        try:
            nuisance = nuisance_fn(
                X, T, Y, fold_spec,
                propensity_model=strategy.propensity_model,
                outcome_model=strategy.outcome_model,
                propensity_trim=strategy.propensity_trim,
                random_state=nuisance_seed,
                outcome_type=self._outcome_type,
            )
        except Exception as e:
            warnings.warn(
                f"Nuisance fitting failed: {e}", ComponentFailureWarning, stacklevel=3
            )
            return None

        # Assemble fit_weights inputs
        surviving = list(oof.keys())
        if fold_spec.n_folds == 1:
            avg_idx = fold_spec.test_indices[0]
            cate_matrix = np.stack(
                [oof[name][avg_idx] for name in surviving], axis=0
            )  # (K, avg_n)
            Y_w = Y[avg_idx]
            T_w = T[avg_idx]
            X_w = X[avg_idx]
            nu_w = NuisanceEstimates(
                e_hat=nuisance.e_hat[avg_idx],
                mu1_hat=nuisance.mu1_hat[avg_idx],
                mu0_hat=nuisance.mu0_hat[avg_idx],
            )
        else:
            cate_matrix = np.stack(
                [oof[name] for name in surviving], axis=0
            )  # (K, n)
            Y_w = Y
            T_w = T
            X_w = X
            nu_w = nuisance

        # Fit weights
        try:
            strategy.fit_weights(cate_matrix, Y_w, T_w, X_w, nu_w)
            if strategy._weights is not None:
                strategy._weights.model_names = surviving
        except Exception as e:
            warnings.warn(
                f"fit_weights failed: {e}", ComponentFailureWarning, stacklevel=3
            )
            return None

        # Retrain surviving adapters on the full dataset
        surviving_methods = [m for m in cate_methods if m.name in set(surviving)]
        fitted, times = self._run_fit_loop(
            surviving_methods, X, T, Y, random_state, n_jobs=n_jobs, **kwargs
        )
        if not fitted:
            return None

        return {
            "fitted_adapters": fitted,
            "component_fit_times": times,
            "oof_cate_predictions": cate_matrix,
            "oof_cate_model_names": surviving,
            "nuisance": nu_w,
            "fit_weights_X": X_w,
            "fit_weights_T": T_w,
            "fit_weights_Y": Y_w,
        }

    # ------------------------------------------------------------------
    # Strategy override
    # ------------------------------------------------------------------

    def _resolve_override(
        self,
        aggregation: str | AggregationStrategy | None,
    ) -> AggregationStrategy:
        """Resolve an aggregation override to a ready-to-use strategy.

        None → return self.aggregation (already fitted, no work done).
        string → resolve via _STRING_FACTORIES.
        PointwiseStrategy → return as-is (stateless).
        AgreementStrategy → compute_weights() using cached training CATE predictions.
        SupervisedStrategy → fit_weights() using cached OOF predictions and nuisance.
            Raises RuntimeError if no OOF artifacts are available (i.e., fit() was
            not called with a SupervisedStrategy).

        The override strategy object is mutated (weights are set on it). Pass a
        fresh instance for each call if you need independent weight objects.
        """
        if aggregation is None:
            return self.aggregation

        if isinstance(aggregation, str):
            aggregation = _STRING_FACTORIES[aggregation]()

        if isinstance(aggregation, PointwiseStrategy):
            return aggregation

        if isinstance(aggregation, AgreementStrategy):
            if not self._cached_cate_model_names:
                raise RuntimeError(
                    "No CATE-capable adapters available for AgreementStrategy override. "
                    "Refit with at least one CATE-capable adapter."
                )
            aggregation.compute_weights(
                self._cached_train_cate_matrix,
                self._cached_cate_model_names,
            )
            return aggregation

        if isinstance(aggregation, SupervisedStrategy):
            if self._cached_oof_cate_predictions is None:
                raise RuntimeError(
                    "No cross-fitting artifacts available. "
                    "Refit with a SupervisedStrategy to use supervised aggregation overrides."
                )
            if isinstance(self.aggregation, SupervisedStrategy):
                self._warn_supervised_config_mismatch(aggregation)
            aggregation.fit_weights(
                self._cached_oof_cate_predictions,
                self._cached_fit_weights_Y,
                self._cached_fit_weights_T,
                self._cached_fit_weights_X,
                self._cached_nuisance,
            )
            if aggregation._weights is not None:
                aggregation._weights.model_names = self._cached_oof_cate_model_names
            return aggregation

        raise TypeError(
            f"aggregation must be a string, PointwiseStrategy, AgreementStrategy, "
            f"or SupervisedStrategy; got {type(aggregation).__name__}"
        )

    def _warn_supervised_config_mismatch(
        self, override: SupervisedStrategy
    ) -> None:
        """Warn if a supervised override has config that differs from the original fit."""
        original = self.aggregation  # known to be SupervisedStrategy
        mismatches = []
        if type(override.split) is not type(original.split):
            mismatches.append("split")
        if override.propensity_trim != original.propensity_trim:
            mismatches.append("propensity_trim")
        # propensity_model and outcome_model are arbitrary objects; comparing
        # them for equality is unreliable. Warn only if the override sets a
        # non-None model where the original used the default (None → LightGBM).
        if override.propensity_model is not None and original.propensity_model is None:
            mismatches.append("propensity_model")
        if override.outcome_model is not None and original.outcome_model is None:
            mismatches.append("outcome_model")
        if mismatches:
            warnings.warn(
                f"Override strategy's {mismatches} configuration is ignored; "
                "using cached artifacts from fit(). "
                "The override's fit_weights() logic is applied to the original "
                "cross-fitting and nuisance estimates.",
                UserWarning,
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # ate / cate
    # ------------------------------------------------------------------

    def ate(
        self,
        X: np.ndarray | None = None,
        aggregation: str | AggregationStrategy | None = None,
    ) -> AteEstimate:
        """Compute the ensemble ATE from fitted component models.

        Parameters:
            X: Covariate matrix for ATE evaluation. If ``None``, uses
                the training covariates from ``fit()``.
            aggregation: Optional override for the aggregation strategy.
                If ``None``, uses the strategy from ``fit()``. See
                ``cate()`` for full override semantics.

        Returns:
            AteEstimate with aggregated ATE and component estimates.
            For bootstrap CIs, use ``bootstrap()``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CausalEnsemble is not fitted. Call fit() first."
            )

        X_eval = X if X is not None else self._X_train
        strategy = self._resolve_override(aggregation)

        component_estimates: dict[str, ComponentAteEstimate] = {}
        for adapter in self._fitted_adapters:
            try:
                component_estimates[adapter.name] = adapter.ate(X_eval)
            except Exception as e:
                warnings.warn(
                    f"Method '{adapter.name}' failed during ate(): {e}",
                    ComponentFailureWarning,
                    stacklevel=2,
                )

        if not component_estimates:
            raise RuntimeError("All component methods failed during ate().")

        if isinstance(strategy, PointwiseStrategy):
            ates = np.array([e.ate for e in component_estimates.values()])
            ate = float(strategy.aggregate(ates))
        else:
            # AgreementStrategy and SupervisedStrategy: ATE = mean of ensemble CATE.
            # Pass the original aggregation parameter so cate() resolves independently
            # (avoids threading a mutated strategy object through two call sites).
            cate_result = self.cate(X_eval, aggregation=aggregation)
            ate = float(cate_result.cate.mean())
            component_estimates = {
                name: ComponentAteEstimate(ate=float(ce.cate.mean()))
                for name, ce in cate_result.component_estimates.items()
            }

        return AteEstimate(
            ate=ate,
            component_estimates=component_estimates,
            aggregation=type(strategy).__name__,
            component_fit_times=self._component_fit_times,
        )

    def cate(
        self,
        X: np.ndarray | None = None,
        aggregation: str | AggregationStrategy | None = None,
    ) -> CateEstimate:
        """Compute ensemble CATE from fitted component models.

        Only methods with ``supports_cate=True`` participate. The aggregation
        strategy determines how component CATE vectors are combined (pointwise
        rule, agreement-based weights, or outcome-supervised weights).

        Parameters:
            X: Covariate matrix for CATE prediction, shape (n, p). If
                ``None``, uses the training covariates from ``fit()``.
            aggregation: Optional override for the aggregation strategy.
                If ``None``, uses the strategy from ``fit()``.
                Accepts a string alias, a PointwiseStrategy (stateless),
                an AgreementStrategy (weights recomputed from cached training
                CATE predictions), or a SupervisedStrategy (weights recomputed
                from cached OOF predictions — requires a prior supervised fit).
                The override object is mutated (weights populated). Pass a
                fresh instance per call if you need independent weight objects.

        Returns:
            CateEstimate with aggregated CATE and component estimates.
            For bootstrap CIs, use ``bootstrap()``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CausalEnsemble is not fitted. Call fit() first."
            )

        X_eval = X if X is not None else self._X_train
        X_eval = np.asarray(X_eval)
        strategy = self._resolve_override(aggregation)

        cate_adapters = [a for a in self._fitted_adapters if a.supports_cate]
        if not cate_adapters:
            raise RuntimeError(
                "No component methods support CATE. Methods available: "
                + ", ".join(a.name for a in self._fitted_adapters)
            )

        component_estimates: dict[str, ComponentCateEstimate] = {}
        for adapter in cate_adapters:
            try:
                component_estimates[adapter.name] = adapter.cate(X_eval)
            except Exception as e:
                warnings.warn(
                    f"Method '{adapter.name}' failed during cate(): {e}",
                    ComponentFailureWarning,
                    stacklevel=2,
                )

        if not component_estimates:
            raise RuntimeError(
                "All CATE-capable component methods failed during cate()."
            )

        cate_matrix = np.stack(
            [e.cate for e in component_estimates.values()], axis=0
        )  # shape (K, n)
        cate = strategy.aggregate(cate_matrix)

        return CateEstimate(
            cate=cate,
            component_estimates=component_estimates,
            aggregation=type(strategy).__name__,
            ensemble_weights=strategy.ensemble_weights,
        )

    # ------------------------------------------------------------------
    # bootstrap
    # ------------------------------------------------------------------

    def bootstrap(
        self,
        X: np.ndarray | None = None,
        n_boot: int = 200,
        alpha: float = 0.05,
        random_state: int | None = None,
        n_jobs: int = 1,
        method: str = "nonparametric",
        subsample_size: float | int = 0.5,
    ) -> BootstrapResult:
        """Bootstrap inference for ensemble ATE and CATE.

        Resamples training data, refits the entire pipeline (component models
        and aggregation weights) per replicate, and computes percentile CIs
        for both ATE and CATE.

        For SupervisedStrategy, each replicate re-runs the full cross-fitting
        and nuisance estimation on the resampled data — this is the honest
        bootstrap but is expensive: B × Q × K adapter fits plus B nuisance
        fits. Use lightweight nuisance models (set propensity_model and
        outcome_model on the strategy) when running many replicates.

        Parameters:
            X: Covariates for prediction. If ``None``, uses X_train.
            n_boot: Number of bootstrap replicates.
            alpha: Significance level (default 0.05 → 95% CIs).
            random_state: Seed for bootstrap resampling. If ``None`` and
                ``fit()`` was called with a seed, bootstrap seeds are
                derived deterministically from the fit seed.
            n_jobs: Parallel workers for bootstrap replicates. 1 =
                sequential, -1 = all cores. Requires joblib. Each
                replicate's inner cross-fitting and nuisance estimation
                run sequentially — bootstrap is treated as the outermost
                parallel level to avoid oversubscription.
            method: Resampling scheme.
                ``"nonparametric"`` (default): n-out-of-n with replacement,
                the standard Efron bootstrap. Replicates contain duplicates
                of the original units, which can leak across cross-fit
                folds inside DML-style components and supervised
                aggregation wrappers, weakening the orthogonality
                argument those components rely on.
                ``"subsample"``: m-out-of-n without replacement, T-stratified.
                Eliminates duplicates so any downstream cross-fit (the
                supervised pipeline's outer loop, every adapter's internal
                cross-fit, supervised aggregation CV) stays honest. CIs
                use the Politis–Romano (1994) scaled-percentile correction:
                ``θ̂ + sqrt(m/n) * percentile(T_m - θ̂)``.
            subsample_size: Subsample size when ``method="subsample"``.
                Float in (0, 1) is interpreted as a fraction of n; int is
                interpreted as m directly. Must satisfy 1 ≤ m < n. The
                default 0.5 (m = n/2) is a practical compromise between
                preserving sample size for the components' nuisance fits
                and giving the subsampling correction enough room to
                work; smaller m is justified theoretically but increases
                replicate failure rates and finite-sample bias of the
                components. Ignored when ``method="nonparametric"``.

        Returns:
            BootstrapResult with ATE and CATE point estimates, CIs, and
            bootstrap distributions.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CausalEnsemble is not fitted. Call fit() first."
            )

        if method not in ("nonparametric", "subsample"):
            raise ValueError(
                f"method must be 'nonparametric' or 'subsample'; got {method!r}"
            )

        X_eval = X if X is not None else self._X_train
        n = len(self._Y_train)
        m = _resolve_subsample_size(n, subsample_size) if method == "subsample" else n

        # Stratification only makes sense for discrete T. Match the
        # threshold and warning category used by aggregation.splitting
        # (n_unique > 10 → continuous-T fallback) so the package speaks
        # with one voice on this question.
        stratify = True
        if method == "subsample":
            n_unique_T = int(np.unique(self._T_train).size)
            if n_unique_T > 10:
                warnings.warn(
                    f"method='subsample' with T having {n_unique_T} unique "
                    f"values suggests continuous treatment. Stratification "
                    f"requires discrete T; falling back to unstratified "
                    f"subsample (no replacement).",
                    UserWarning,
                    stacklevel=2,
                )
                stratify = False

        # Stash for replicate methods (read by _single_bootstrap and
        # _supervised_bootstrap_replicate without changing their tuple
        # protocol with the parallel chunk dispatcher).
        self._boot_method = method
        self._boot_m = m
        self._boot_stratify = stratify

        # Point estimates from the original fit
        ate_result = self.ate(X_eval)
        has_cate = any(a.supports_cate for a in self._fitted_adapters)
        cate_result = self.cate(X_eval) if has_cate else None

        comp_ate_ests = dict(ate_result.component_estimates)
        comp_cate_ests = (
            dict(cate_result.component_estimates) if cate_result is not None else None
        )

        rng = self._get_boot_rng(random_state)
        seeds = rng.integers(0, 2**32 - 1, size=n_boot).tolist()

        if n_jobs == 1:
            raw = [self._single_bootstrap(X_eval, s) for s in seeds]
        else:
            import os

            n_workers = n_jobs if n_jobs > 0 else os.cpu_count()
            n_chunks = min(n_workers, n_boot)
            chunks = [c for c in np.array_split(seeds, n_chunks) if len(c) > 0]
            chunk_tasks = [(X_eval, list(chunk)) for chunk in chunks]
            chunk_results = parallel_map(
                n_jobs, self._bootstrap_chunk, chunk_tasks
            )
            raw = [r for chunk in chunk_results for r in chunk]

        valid = [r for r in raw if r is not None]
        n_failed = n_boot - len(valid)

        if n_failed > 0 and valid:
            fail_pct = n_failed / n_boot * 100
            if fail_pct > 10:
                warnings.warn(
                    f"{n_failed}/{n_boot} bootstrap replicates failed "
                    f"({fail_pct:.0f}%); CIs may be unreliable.",
                    BootstrapWarning,
                    stacklevel=2,
                )

        if not valid:
            warnings.warn(
                "All bootstrap replicates failed; CIs not computed.",
                BootstrapWarning,
                stacklevel=2,
            )
            return BootstrapResult(
                ate=ate_result.ate,
                ate_ci_lower=float("nan"),
                ate_ci_upper=float("nan"),
                boot_ates=np.array([]),
                cate=cate_result.cate if cate_result is not None else None,
                cate_ci_lower=None,
                cate_ci_upper=None,
                boot_cates=None,
                component_boot_ates={},
                component_ate_estimates=comp_ate_ests,
                component_cate_estimates=comp_cate_ests,
                n_boot=n_boot,
                n_failed=n_failed,
                alpha=alpha,
                aggregation=type(self.aggregation).__name__,
                ensemble_weights=self.aggregation.ensemble_weights,
                method=method,
                subsample_m=m if method == "subsample" else None,
            )

        boot_ates = np.array([r[0] for r in valid])
        boot_comp: dict[str, list[float]] = {}
        for _, comp_ates, _ in valid:
            for name, a in comp_ates.items():
                boot_comp.setdefault(name, []).append(a)

        # CI construction. Centered scaled-percentile (Politis–Romano):
        # CI = θ̂ + scale * percentile(T_b - θ̂), with scale = sqrt(m/n).
        # When method="nonparametric" we have m == n, so scale == 1 and the
        # expression reduces algebraically to the standard percentile
        # bootstrap — the centering/recentering cancels out.
        scale = float(np.sqrt(m / n)) if method == "subsample" else 1.0
        theta_hat = float(ate_result.ate)
        centered_ates = boot_ates - theta_hat
        ate_ci_lower = float(
            theta_hat + scale * np.percentile(centered_ates, 100 * alpha / 2)
        )
        ate_ci_upper = float(
            theta_hat + scale * np.percentile(centered_ates, 100 * (1 - alpha / 2))
        )

        component_boot_ates = {k: np.array(v) for k, v in boot_comp.items()}

        # CATE bootstrap
        replicate_cates = [r[2] for r in valid if r[2] is not None]
        if replicate_cates:
            boot_cates = np.stack(replicate_cates, axis=0)  # (B, n)
            cate_point = cate_result.cate  # shape (n_eval,)
            centered_cates = boot_cates - cate_point[None, :]
            cate_ci_lower = cate_point + scale * np.percentile(
                centered_cates, 100 * alpha / 2, axis=0
            )
            cate_ci_upper = cate_point + scale * np.percentile(
                centered_cates, 100 * (1 - alpha / 2), axis=0
            )
        else:
            boot_cates = None
            cate_ci_lower = None
            cate_ci_upper = None

        # Containment is governed by the same condition for both schemes: the
        # sqrt(m/n) scaling changes the CI width but never moves an endpoint
        # across theta_hat, so theta_hat ∈ CI iff it lies in the central
        # 1−alpha percentile range of the replicate ATEs. Flag exclusion under
        # either scheme; only the remedy text is scheme-specific.
        if not (ate_ci_lower <= theta_hat <= ate_ci_upper):
            if method == "subsample":
                remedy = (
                    "This typically indicates finite-sample bias of one or "
                    "more components at the subsample size — consider "
                    "increasing subsample_size or inspecting per-component "
                    "bootstrap distributions."
                )
            else:  # nonparametric
                remedy = (
                    "The reduced effective sample size of with-replacement "
                    "resampling (~63% distinct units) or per-replicate weight "
                    "re-optimization can shift the replicate distribution "
                    "relative to the point estimate — consider the subsample "
                    "scheme or inspecting per-component bootstrap distributions."
                )
            warnings.warn(
                f"{method.capitalize()} bootstrap ATE CI [{ate_ci_lower:.4g}, "
                f"{ate_ci_upper:.4g}] does not contain the point estimate "
                f"{theta_hat:.4g} (m={m}, n={n}). {remedy}",
                BootstrapWarning,
                stacklevel=2,
            )

        return BootstrapResult(
            ate=ate_result.ate,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            boot_ates=boot_ates,
            cate=cate_result.cate if cate_result is not None else None,
            cate_ci_lower=cate_ci_lower,
            cate_ci_upper=cate_ci_upper,
            boot_cates=boot_cates,
            component_boot_ates=component_boot_ates,
            component_ate_estimates=comp_ate_ests,
            component_cate_estimates=comp_cate_ests,
            n_boot=n_boot,
            n_failed=n_failed,
            alpha=alpha,
            aggregation=type(self.aggregation).__name__,
            ensemble_weights=self.aggregation.ensemble_weights,
            method=method,
            subsample_m=m if method == "subsample" else None,
        )

    # ------------------------------------------------------------------
    # estimate (convenience wrapper)
    # ------------------------------------------------------------------

    def estimate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        n_boot: int = 0,
        alpha: float = 0.05,
        random_state: int | None = None,
        n_jobs: int = 1,
        method: str = "nonparametric",
        subsample_size: float | int = 0.5,
        **kwargs,
    ) -> AteEstimate | BootstrapResult:
        """Fit and compute ATE (and optionally bootstrap CIs) in one call.

        Parameters:
            X: Covariate matrix, shape (n, p).
            T: Treatment vector, shape (n,).
            Y: Outcome vector, shape (n,).
            n_boot: Number of bootstrap replicates. 0 = point estimate only.
            alpha: Significance level for bootstrap CIs (default 0.05 → 95%).
            random_state: Seed for reproducibility.
            n_jobs: Parallel workers. Routed to the outermost active
                level: when ``n_boot > 0``, bootstrap is parallelized and
                the inner ``fit()`` runs sequentially; otherwise the
                supervised cross-fitting / component-fit loop is
                parallelized.
            method: Bootstrap resampling scheme; see :meth:`bootstrap`.
            subsample_size: Subsample size when ``method="subsample"``;
                see :meth:`bootstrap`.
            **kwargs: Forwarded to each method's ``fit()`` call.

        Returns:
            :class:`AteEstimate` when ``n_boot=0``,
            :class:`BootstrapResult` when ``n_boot > 0``.
        """
        fit_n_jobs = 1 if n_boot > 0 else n_jobs
        self.fit(X, T, Y, random_state=random_state, n_jobs=fit_n_jobs, **kwargs)
        if n_boot > 0:
            return self.bootstrap(
                n_boot=n_boot,
                alpha=alpha,
                random_state=random_state,
                n_jobs=n_jobs,
                method=method,
                subsample_size=subsample_size,
            )
        return self.ate()

    # ------------------------------------------------------------------
    # visualization (thin wrappers over metacausal.plots)
    # ------------------------------------------------------------------

    def weights(
        self,
        *,
        ax: Axes | None = None,
        on_uniform: Literal["warn", "error", "ignore"] = "warn",
        sort: bool = True,
    ) -> Axes:
        """Bar chart of the fitted aggregation strategy's weights.

        Thin wrapper around :func:`metacausal.plots.weights`; see there
        for the full parameter reference. Requires the ``plots`` extra
        (``pip install 'metacausal[plots]'``). Raises ``ValueError`` if
        the fitted aggregation strategy is pointwise (no weights to plot).

        Returns:
            matplotlib Axes the plot was drawn on.
        """
        from metacausal.plots import weights as _weights

        return _weights(self, ax=ax, on_uniform=on_uniform, sort=sort)

    def disagreement(
        self,
        X: np.ndarray,
        *,
        ax: Axes | None = None,
        metric: Literal["spearman", "pearson", "rmse"] = "spearman",
        cluster: bool = False,
        annotate: bool = True,
    ) -> Axes:
        """Pairwise disagreement heatmap between component CATEs on ``X``.

        Thin wrapper around :func:`metacausal.plots.disagreement`; see
        there for the full parameter reference. Requires the ``plots``
        extra (``pip install 'metacausal[plots]'``).

        Returns:
            matplotlib Axes the plot was drawn on.
        """
        from metacausal.plots import disagreement as _disagreement

        return _disagreement(self, X, ax=ax, metric=metric, cluster=cluster, annotate=annotate)

    # ------------------------------------------------------------------
    # bootstrap internals
    # ------------------------------------------------------------------

    def _get_boot_rng(self, random_state: int | None) -> np.random.Generator:
        """Get an RNG for bootstrap resampling.

        Deterministic and idempotent: calling this method twice with the
        same arguments produces identical RNG states.
        """
        if random_state is not None:
            return np.random.default_rng(random_state)
        if self._fit_random_state is not None:
            rng = np.random.default_rng(self._fit_random_state)
            rng.integers(0, 2**32 - 1, size=len(self._wrapped_methods))
            return rng
        return np.random.default_rng(None)

    def _supervised_bootstrap_replicate(
        self,
        X_eval: np.ndarray,
        idx: np.ndarray,
        seed: int,
    ) -> tuple[float, dict[str, float], np.ndarray | None] | None:
        """One supervised bootstrap replicate on resampled training data.

        Re-runs the full supervised pipeline (cross-fitting, nuisance,
        weight optimisation, retrain) on the resampled data, then predicts
        on X_eval. Returns None on any failure (counted as n_failed).
        """
        X_boot = self._X_train[idx]
        T_boot = self._T_train[idx]
        Y_boot = self._Y_train[idx]

        # Fresh strategy copy so _weights are not shared across replicates
        boot_strategy = copy.deepcopy(self.aggregation)

        # Bootstrap is the outer level — force inner cross-fitting sequential.
        result = self._run_supervised_pipeline(
            X_boot, T_boot, Y_boot, seed, boot_strategy,
            n_jobs=1, **self._fit_kwargs,
        )
        if result is None:
            return None

        surviving = result["oof_cate_model_names"]
        fitted = result["fitted_adapters"]

        # Predict component CATEs on X_eval
        component_cate_ests: dict[str, ComponentCateEstimate] = {}
        for adapter in fitted:
            try:
                component_cate_ests[adapter.name] = adapter.cate(X_eval)
            except Exception:
                pass

        if not component_cate_ests:
            return None

        # cate_matrix rows must align with surviving (= model_names order)
        pred_names = [n for n in surviving if n in component_cate_ests]
        if len(pred_names) != len(surviving):
            # Some adapters failed prediction — weight alignment broken
            return None

        try:
            cate_matrix = np.stack(
                [component_cate_ests[n].cate for n in pred_names], axis=0
            )
            ensemble_cate = boot_strategy.aggregate(cate_matrix)
        except Exception:
            return None

        ensemble_ate = float(ensemble_cate.mean())
        comp_ates = {
            n: float(component_cate_ests[n].cate.mean())
            for n in component_cate_ests
        }
        return ensemble_ate, comp_ates, ensemble_cate

    def _single_bootstrap(
        self,
        X_eval: np.ndarray,
        seed: int,
    ) -> tuple[float, dict[str, float], np.ndarray | None] | None:
        """Run one bootstrap replicate.

        Returns (ensemble_ate, component_ates_dict, ensemble_cate_or_None)
        or None on total failure.
        """
        rng = np.random.default_rng(seed)
        n = len(self._Y_train)
        if getattr(self, "_boot_method", "nonparametric") == "subsample":
            if getattr(self, "_boot_stratify", True):
                idx = _stratified_subsample(rng, self._T_train, self._boot_m)
            else:
                idx = rng.choice(n, size=self._boot_m, replace=False)
        else:
            idx = rng.choice(n, size=n, replace=True)

        if isinstance(self.aggregation, SupervisedStrategy):
            return self._supervised_bootstrap_replicate(X_eval, idx, seed)

        method_seed_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))

        component_ate_estimates: dict[str, ComponentAteEstimate] = {}
        component_cate_estimates: dict[str, ComponentCateEstimate] = {}
        # For AgreementStrategy: CATE predictions on the resampled training data
        component_train_cate_estimates: dict[str, ComponentCateEstimate] = {}

        for m in self._wrapped_methods:
            mseed = int(method_seed_rng.integers(0, 2**32 - 1))
            try:
                adapter = copy.deepcopy(m)
                adapter.fit(
                    self._X_train[idx],
                    self._T_train[idx],
                    self._Y_train[idx],
                    random_state=mseed,
                    **self._fit_kwargs,
                )
                component_ate_estimates[adapter.name] = adapter.ate(X_eval)
                if adapter.supports_cate:
                    component_cate_estimates[adapter.name] = adapter.cate(X_eval)
                    if isinstance(self.aggregation, AgreementStrategy):
                        component_train_cate_estimates[adapter.name] = adapter.cate(
                            self._X_train[idx]
                        )
            except Exception:
                continue

        if not component_ate_estimates:
            return None

        ensemble_cate: np.ndarray | None = None

        if isinstance(self.aggregation, AgreementStrategy):
            # Recompute weights from this replicate's training CATE predictions,
            # then apply to X_eval CATE predictions.
            if not component_train_cate_estimates:
                return None
            common_names = [
                k for k in component_train_cate_estimates
                if k in component_cate_estimates
            ]
            if not common_names:
                return None
            train_cate_matrix = np.stack(
                [component_train_cate_estimates[k].cate for k in common_names], axis=0
            )
            boot_strategy = copy.deepcopy(self.aggregation)
            try:
                boot_strategy.compute_weights(train_cate_matrix, common_names)
            except (ValueError, RuntimeError):
                return None
            cate_matrix = np.stack(
                [component_cate_estimates[k].cate for k in common_names], axis=0
            )
            ensemble_cate = boot_strategy.aggregate(cate_matrix)
            if ensemble_cate is None:
                return None
            ensemble_ate = float(ensemble_cate.mean())
            comp_ates = {
                k: float(v.cate.mean()) for k, v in component_cate_estimates.items()
            }
        else:
            ates = np.array([e.ate for e in component_ate_estimates.values()])
            ensemble_ate = float(self.aggregation.aggregate(ates))
            comp_ates = {k: v.ate for k, v in component_ate_estimates.items()}
            if component_cate_estimates:
                cate_matrix = np.stack(
                    [e.cate for e in component_cate_estimates.values()], axis=0
                )
                ensemble_cate = self.aggregation.aggregate(cate_matrix)

        return ensemble_ate, comp_ates, ensemble_cate

    def _bootstrap_chunk(
        self,
        X_eval: np.ndarray,
        seeds: list[int],
    ) -> list[tuple[float, dict[str, float], np.ndarray | None] | None]:
        """Run a batch of bootstrap replicates (one chunk per worker)."""
        return [self._single_bootstrap(X_eval, s) for s in seeds]
