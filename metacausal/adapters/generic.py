"""Generic adapters for callables and simple estimators."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


class GenericATEAdapter:
    """Wrap a callable ``fn(X, T, Y) -> float`` as an ATE-only causal estimator.

    Parameters:
        fn: A function that takes ``(X, T, Y)`` and returns a scalar ATE
            or a :class:`ComponentAteEstimate`.
        name: Display name for this estimator. Default ``"custom"``.

    Examples:
        >>> from metacausal.adapters import GenericATEAdapter
        >>> from metacausal.datasets import load_lalonde
        >>> X, T, Y = load_lalonde()
        >>> def naive_diff(X, T, Y):
        ...     return float(Y[T == 1].mean() - Y[T == 0].mean())
        >>> adapter = GenericATEAdapter(naive_diff, name="naive_diff")
        >>> adapter.fit(X, T, Y)
        >>> adapter.ate()
        ComponentAteEstimate(ate=...)
    """

    def __init__(
        self,
        fn: Callable,
        name: str = "custom",
        supported_outcome_types: tuple[str, ...] = ("continuous",),
    ) -> None:
        self._fn = fn
        self._name = name
        self.supported_outcome_types = tuple(supported_outcome_types)
        self._X: np.ndarray | None = None
        self._T: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def validate_outcome_type(self, detected: str) -> None:
        """No-op: the wrapped callable is opaque, so the user's
        ``supported_outcome_types`` declaration is the contract.
        """
        return

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        **kwargs,
    ) -> None:
        self._X = X
        self._T = T
        self._Y = Y
        self._is_fitted = True

    def ate(self, X: np.ndarray | None = None) -> ComponentAteEstimate:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self._name} is not fitted. Call fit() first."
            )
        result = self._fn(self._X, self._T, self._Y)
        if isinstance(result, ComponentAteEstimate):
            return result
        return ComponentAteEstimate(ate=float(result))

    def cate(self, X: np.ndarray) -> ComponentCateEstimate:
        raise NotImplementedError(
            f"{self._name} does not support CATE estimation."
        )


# Backward-compatible alias
GenericAdapter = GenericATEAdapter


class GenericCATEAdapter:
    """Wrap callables as a CATE-capable causal estimator.

    Reduces the barrier for prototyping custom CATE estimators to two or three
    functions — one for fitting, one for CATE prediction, and an optional one
    for ATE — without implementing the full ``CausalEstimator`` protocol.

    Parameters
    ----------
    fn_fit : callable
        ``fn_fit(X, T, Y, **kwargs) -> state``

        Called during ``fit()``. Receives the training arrays and any keyword
        arguments forwarded by ``CausalEnsemble`` (e.g. ``random_state``).
        Returns an opaque *state* object (fitted model, dict, namedtuple,
        ``None`` for stateless callables …) passed to ``fn_cate`` and
        ``fn_ate`` at prediction time.

        **Reproducibility:** if ``fn_fit`` uses randomness, it is the caller's
        responsibility to consume ``kwargs.get("random_state")``.
        ``CausalEnsemble`` forwards the seed but cannot enforce its use.

        **Deep-copy requirement:** ``state`` must survive
        ``copy.deepcopy()`` so that bootstrap resampling and supervised
        cross-fitting work correctly. Closures over PyTorch tensors, open file
        handles, or GPU state may fail; store a picklable specification
        (CPU weights, a file path, or constructor arguments) and rebuild
        such objects lazily inside ``fn_cate`` / ``fn_ate`` instead.

    fn_cate : callable
        ``fn_cate(state, X) -> array-like of shape (n,)``

        Called during ``cate()``. Receives the fitted *state* and an
        evaluation covariate matrix ``X`` of shape ``(n, p)``. Must return a
        1-D array (or ``(n, 1)`` array, which is automatically squeezed) of
        CATE predictions.

    fn_ate : callable or None
        ``fn_ate(state, X) -> float | ComponentAteEstimate``

        Optional. Called during ``ate()``. Receives the fitted *state* and
        the evaluation covariate matrix ``X``. Returns either a scalar ATE or
        a :class:`ComponentAteEstimate` (useful when the estimator provides
        a native, more efficient ATE — e.g. a DR/AIPW mean rather than
        ``mean(cate(X))``).

        If ``None`` (default), ``ate()`` falls back to ``mean(cate(X_eval))``.

    name : str
        Display name used in ``CateEstimate.component_estimates`` and warning
        messages. Default ``"custom_cate"``.

    Notes
    -----
    **Parallelism cooperation contract.** MetaCausal automatically suppresses
    the internal joblib-parallel knobs (``n_jobs``, ``cv_n_jobs``, ...) of the
    estimators it wraps for EconML, CausalML, DoubleML, and stochtree whenever
    they run inside one of MetaCausal's own workers (``fit``/``bootstrap``, or
    supervised cross-fitting) -- this stops the wrapped estimator's own
    default from nesting a second worker pool inside the outer one and
    oversubscribing cores. That automatic suppression cannot reach into an
    opaque ``fn_fit``/``fn_cate``/``fn_ate`` callable, so if your model
    parallelizes internally, you are responsible for one of:

    - Constructing it with its own parallelism knob fixed at ``1`` up front,
      before passing it to ``fn_fit``; or
    - Cooperating with the same signal MetaCausal's built-in adapters use:
      inside ``fn_fit``, check
      ``os.environ.get(metacausal.adapters.INNER_WORKER_ENV)`` and pin your
      model to serial when it is set (meaning you are already running inside
      a MetaCausal worker, so any further internal parallelism would nest
      and oversubscribe).

    See "Extending MetaCausal" in the project README for a worked example of
    the second option.

    Examples
    --------
    Wrap a T-learner whose native ATE is a doubly robust (AIPW) mean computed
    at fit time, where the observed ``T`` and ``Y`` are available. This is the
    reason ``fn_ate`` exists: it returns an estimate the default
    ``mean(cate(X))`` fallback cannot produce, because the fallback never sees
    the outcomes.

    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from metacausal import CausalEnsemble
    >>> from metacausal.adapters import GenericCATEAdapter
    >>> from metacausal.datasets import load_lalonde
    >>> X, T, Y = load_lalonde()
    >>> def fit_fn(X, T, Y, **kwargs):
    ...     treated = T == 1
    ...     m1 = LinearRegression().fit(X[treated], Y[treated])
    ...     m0 = LinearRegression().fit(X[~treated], Y[~treated])
    ...     e = LogisticRegression(max_iter=1000).fit(X, T).predict_proba(X)[:, 1]
    ...     mu1, mu0 = m1.predict(X), m0.predict(X)
    ...     # AIPW / doubly robust ATE on the training sample
    ...     ate_dr = float(np.mean(
    ...         (mu1 - mu0)
    ...         + T * (Y - mu1) / e
    ...         - (1 - T) * (Y - mu0) / (1 - e)
    ...     ))
    ...     return {"m1": m1, "m0": m0, "ate_dr": ate_dr}
    >>> def cate_fn(state, X):
    ...     return state["m1"].predict(X) - state["m0"].predict(X)
    >>> def ate_fn(state, X):
    ...     # native DR ATE precomputed at fit time; differs from mean(cate(X))
    ...     return state["ate_dr"]
    >>> adapter = GenericCATEAdapter(fit_fn, cate_fn, fn_ate=ate_fn,
    ...                              name="t_learner")
    >>> ens = CausalEnsemble(methods=[adapter], aggregation="median")
    >>> _ = ens.fit(X, T, Y, random_state=42)
    >>> ens.ate()
    AteEstimate(ate=..., n_methods=1, aggregation='Median', spread=...)
    """

    def __init__(
        self,
        fn_fit: Callable,
        fn_cate: Callable,
        fn_ate: Callable | None = None,
        name: str = "custom_cate",
        supported_outcome_types: tuple[str, ...] = ("continuous",),
    ) -> None:
        self._fn_fit = fn_fit
        self._fn_cate = fn_cate
        self._fn_ate = fn_ate
        self._name = name
        self.supported_outcome_types = tuple(supported_outcome_types)
        self._state: Any = None
        self._X_train: np.ndarray | None = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def validate_outcome_type(self, detected: str) -> None:
        """No-op: the wrapped callables are opaque, so the user's
        ``supported_outcome_types`` declaration is the contract.
        """
        return

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit by calling ``fn_fit(X, T, Y, **kwargs)`` and storing the state."""
        self._state = self._fn_fit(X, T, Y, **kwargs)
        self._X_train = np.asarray(X)
        self._is_fitted = True

    def cate(self, X: np.ndarray) -> ComponentCateEstimate:
        """Return CATE predictions by calling ``fn_cate(state, X)``.

        Accepts ``(n,)`` or ``(n, 1)`` output; raises ``ValueError`` otherwise.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"'{self._name}' is not fitted. Call fit() first."
            )
        raw = np.asarray(
            self._fn_cate(self._state, np.asarray(X)), dtype=float
        )
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 1:
            raise ValueError(
                f"fn_cate for '{self._name}' must return shape (n,) or (n, 1), "
                f"got {raw.shape}"
            )
        return ComponentCateEstimate(cate=raw)

    def ate(self, X: np.ndarray | None = None) -> ComponentAteEstimate:
        """Return the ATE.

        If ``fn_ate`` was provided, calls ``fn_ate(state, X_eval)``.
        Otherwise returns ``mean(cate(X_eval))``.

        ``X_eval`` is ``X`` if given, otherwise the training data from
        ``fit()``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"'{self._name}' is not fitted. Call fit() first."
            )
        X_eval = np.asarray(X) if X is not None else self._X_train

        if self._fn_ate is not None:
            result = self._fn_ate(self._state, X_eval)
            if isinstance(result, ComponentAteEstimate):
                return result
            return ComponentAteEstimate(ate=float(result))

        return ComponentAteEstimate(ate=float(self.cate(X_eval).cate.mean()))
