"""Adapter for EconML estimators."""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np

from metacausal._parallel import INNER_WORKER_ENV, force_serial
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


def _is_econml(obj) -> bool:
    """Check if an object is an EconML estimator (without importing econml)."""
    module = type(obj).__module__ or ""
    return module.startswith("econml")


class EconMLAdapter:
    """Wrap an initialized-but-unfitted EconML estimator.

    Supports any EconML estimator with ``.fit(Y, T, X=X)`` and
    ``.ate(X=X)`` or ``.effect(X=X)`` methods. The estimator template
    is deep-copied during ``fit()`` so that the original remains
    reusable for bootstrap resampling.

    Parameters:
        model: An initialized (unfitted) EconML estimator instance.
        name: Display name. Defaults to the class name.
        alpha: Significance level for analytical confidence intervals.
            Passed through to ``ate_interval()`` / ``effect_interval()``.
        inference: Optional EconML fit-time inference backend, forwarded
            to ``model.fit(..., inference=inference)``.

    Examples:
        >>> from econml.dml import LinearDML
        >>> from metacausal.adapters import EconMLAdapter
        >>> from metacausal.datasets import load_lalonde
        >>> from sklearn.ensemble import (
        ...     HistGradientBoostingRegressor as HGBR,
        ...     HistGradientBoostingClassifier as HGBC,
        ... )
        >>> X, T, Y = load_lalonde()
        >>> dml = EconMLAdapter(
        ...     LinearDML(
        ...         model_y=HGBR(max_iter=20), model_t=HGBC(max_iter=20),
        ...         discrete_treatment=True, cv=2,
        ...     ),
        ... )
        >>> dml.fit(X, T, Y, random_state=42)
        >>> result = dml.ate()
        >>> result
        ComponentAteEstimate(ate=..., ci=[..., ...])
    """

    def __init__(
        self,
        model,
        name: str | None = None,
        *,
        alpha: float = 0.05,
        inference: Any = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._model = model
        self._name = name or type(model).__name__
        self.alpha = alpha
        self.inference = inference
        self.supported_outcome_types = self._infer_supported_types(model)
        self._fitted_model = None
        self._X_train: np.ndarray | None = None
        self._is_fitted = False

    # EconML classes that natively support binary outcomes via a
    # ``discrete_outcome`` constructor flag (econml >= 0.16). Mapped to
    # the constructor slot that must be a classifier in the binary case.
    _BINARY_CAPABLE_SLOTS = {
        "CausalForestDML": "model_y",
        "DRLearner": "model_regression",
    }

    @classmethod
    def _infer_supported_types(cls, model) -> tuple[str, ...]:
        """EconML's binary-outcome surface is uneven across classes.
        Only the ones in :attr:`_BINARY_CAPABLE_SLOTS` accept a
        classifier outcome model and a ``discrete_outcome=True`` flag;
        others (S/T/X-learners) call ``.predict()`` directly and would
        return hard labels for a classifier.
        """
        if type(model).__name__ in cls._BINARY_CAPABLE_SLOTS:
            return ("continuous", "binary")
        return ("continuous",)

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def validate_outcome_type(self, detected: str) -> None:
        """For binary outcomes, the wrapped EconML estimator must have
        ``discrete_outcome=True`` and a classifier in the slot that
        models the conditional outcome (``model_y`` for
        :class:`CausalForestDML`, ``model_regression`` for
        :class:`DRLearner`).
        """
        if detected != "binary":
            return
        cls_name = type(self._model).__name__
        slot = self._BINARY_CAPABLE_SLOTS.get(cls_name)
        if slot is None:
            return
        if not getattr(self._model, "discrete_outcome", False):
            raise ValueError(
                f"{self._name}: with a binary outcome, {cls_name} must be "
                f"constructed with discrete_outcome=True."
            )
        learner = getattr(self._model, slot, None)
        if learner is not None and not hasattr(learner, "predict_proba"):
            raise ValueError(
                f"{self._name}: with a binary outcome, {slot} must be a "
                f"classifier (must implement predict_proba). Got "
                f"{type(learner).__name__!r}."
            )

    # Attributes on EconML estimators that hold nested sub-models.
    _SUB_MODEL_ATTRS = (
        "model_y", "model_t", "model_final",
        "model_regression", "model_propensity",
        "models", "overall_model", "propensity_model",
    )

    @staticmethod
    @contextmanager
    def _suppress_force_all_finite_futurewarning():
        """Suppress sklearn's renamed-parameter warning on upstream
        EconML prediction and inference calls only.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"'force_all_finite' was renamed to 'ensure_all_finite'.*",
                module=r"sklearn\.utils\.deprecation",
            )
            yield

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        **kwargs,
    ) -> None:
        import copy
        from sklearn.exceptions import DataConversionWarning

        random_state = kwargs.pop("random_state", None)
        inference = kwargs.pop("inference", self.inference)

        model = copy.deepcopy(self._model)
        if random_state is not None:
            self._seed_model(model, random_state)
        # When fitting inside one of MetaCausal's own parallel workers
        # (bootstrap replicate, parallel component fit, or supervised
        # cross-fit), pin the wrapped estimator's joblib parallelism to a
        # single job. EconML's CausalForestDML defaults to n_jobs=-1, whose
        # inner loky pool would otherwise nest inside this worker and
        # oversubscribe cores, intermittently segfaulting EconML's Cython
        # tree builder. The thread-pin env vars do not reach joblib's n_jobs,
        # so it must be pinned explicitly. Outside a worker (serial outer
        # path) the estimator keeps its own n_jobs.
        if os.environ.get(INNER_WORKER_ENV):
            self._force_serial(model)
        # Surgical suppression of an upstream econml/sklearn interaction:
        # when an EconML estimator is configured with ``discrete_outcome=True``
        # (e.g., the binary-pool DRLearner), econml's _OrthoLearner reshapes
        # Y to a column vector (econml/_ortho_learner.py: ``Y = ...reshape(-1, 1)``)
        # before passing it to the inner sklearn classifier; sklearn then fires
        # ``DataConversionWarning`` once per fold to hint that 1D would have
        # done. Functionally harmless (sklearn calls ``column_or_1d(...)``
        # internally and proceeds). The filter below is narrowed by category,
        # message regex, and source module so any *other* DataConversionWarning
        # — different message, different source — passes through unaffected,
        # and the filter scope is bounded to this single ``fit`` call.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DataConversionWarning,
                message=r"A column-vector y was passed when a 1d array was expected.*",
                module=r"sklearn\.utils\.validation",
            )
            if inference is None:
                model.fit(Y, T, X=X)
            else:
                model.fit(Y, T, X=X, inference=inference)

        self._fitted_model = model
        self._X_train = X
        self._is_fitted = True

    def ate(self, X: np.ndarray | None = None) -> ComponentAteEstimate:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self._name} is not fitted. Call fit() first."
            )

        model = self._fitted_model
        X_eval = X if X is not None else self._X_train

        # EconML returns 0-d arrays for continuous outcomes but 1-d
        # ``(1,)`` arrays when ``discrete_outcome=True`` (one entry per
        # non-baseline outcome class). NumPy 2.x rejects ``float(...)``
        # on 1-d arrays, so flatten before scalar conversion.
        def _scalar(x) -> float:
            return float(np.asarray(x).ravel()[0])

        # EconML estimators expose ate() or effect()
        with self._suppress_force_all_finite_futurewarning():
            if hasattr(model, "ate"):
                ate = _scalar(model.ate(X=X_eval))
            elif hasattr(model, "effect"):
                ate = float(np.mean(model.effect(X=X_eval)))
            else:
                raise AttributeError(
                    f"{self._name} has neither .ate() nor .effect() method"
                )

        # Extract CI if available
        ci_lower, ci_upper = None, None
        if hasattr(model, "ate_interval"):
            try:
                with self._suppress_force_all_finite_futurewarning():
                    lo, hi = model.ate_interval(X=X_eval, alpha=self.alpha)
                ci_lower, ci_upper = _scalar(lo), _scalar(hi)
            except Exception:
                pass
        elif hasattr(model, "effect_interval"):
            try:
                with self._suppress_force_all_finite_futurewarning():
                    lo, hi = model.effect_interval(X=X_eval, alpha=self.alpha)
                ci_lower = float(np.mean(lo))
                ci_upper = float(np.mean(hi))
            except Exception:
                pass

        return ComponentAteEstimate(
            ate=ate, ci_lower=ci_lower, ci_upper=ci_upper
        )

    def cate(self, X: np.ndarray) -> ComponentCateEstimate:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self._name} is not fitted. Call fit() first."
            )

        model = self._fitted_model
        with self._suppress_force_all_finite_futurewarning():
            cate = np.asarray(model.effect(X=X), dtype=np.float64).ravel()

        # Extract per-observation CIs if available
        ci_lower, ci_upper = None, None
        if hasattr(model, "effect_interval"):
            try:
                with self._suppress_force_all_finite_futurewarning():
                    lo, hi = model.effect_interval(X=X, alpha=self.alpha)
                ci_lower = np.asarray(lo, dtype=np.float64).ravel()
                ci_upper = np.asarray(hi, dtype=np.float64).ravel()
            except Exception:
                pass

        return ComponentCateEstimate(
            cate=cate, ci_lower=ci_lower, ci_upper=ci_upper
        )

    @classmethod
    def _seed_model(cls, model, random_state: int) -> None:
        """Set random_state on *model* and its nested sub-estimators."""
        rng = np.random.default_rng(random_state)
        if hasattr(model, "random_state"):
            model.random_state = int(rng.integers(0, 2**32 - 1))
        for attr in cls._SUB_MODEL_ATTRS:
            sub = getattr(model, attr, None)
            if sub is None:
                continue
            if isinstance(sub, list):
                for item in sub:
                    if hasattr(item, "random_state"):
                        item.random_state = int(rng.integers(0, 2**32 - 1))
            elif hasattr(sub, "random_state"):
                sub.random_state = int(rng.integers(0, 2**32 - 1))

    @classmethod
    def _force_serial(cls, model) -> None:
        """Set ``n_jobs=1`` on *model* and its nested sub-estimators.

        Called when fitting inside a MetaCausal worker (detected via the
        :data:`~metacausal._parallel.INNER_WORKER_ENV` sentinel). Prevents a
        joblib-parallel component -- EconML's ``CausalForestDML``, or a
        user-supplied nuisance learner such as a ``RandomForest`` with
        ``n_jobs=-1`` -- from nesting a second worker pool inside the outer
        one.         Mirrors :meth:`_seed_model`'s traversal so list-valued slots
        (e.g. ``TLearner.models``) are covered too. Delegates the actual
        pinning to the shared :func:`~metacausal._parallel.force_serial`
        helper, scoped to EconML's known nested-model slots.
        """
        force_serial(model, "n_jobs", cls._SUB_MODEL_ATTRS)
