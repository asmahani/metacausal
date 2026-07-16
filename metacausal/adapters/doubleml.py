"""Adapter for DoubleML estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


class DoubleMLAdapter:
    """Wrap a DoubleML estimator class for use in CausalEnsemble.

    DoubleML bundles data and model at construction time, so this adapter
    accepts the **class** and nuisance model configuration (not an
    initialized object). A fresh DoubleML object is constructed and fitted
    on each call to ``fit()``, which is essential for bootstrap resampling.

    Parameters:
        model_class: A DoubleML class, e.g. ``DoubleMLIRM`` or ``DoubleMLPLR``.
        name: Display name. Defaults to the class name.
        alpha: Significance level for analytical ATE confidence intervals.
            Passed through to ``DoubleML.confint(level=1 - alpha)``.
        **kwargs: Keyword arguments forwarded to the DoubleML constructor
            (excluding ``obj_dml_data``). Typically includes ``ml_l``,
            ``ml_m``, ``ml_g``, ``n_folds``, etc.

    Examples:
        >>> from metacausal.adapters import DoubleMLAdapter
        >>> from doubleml import DoubleMLIRM
        >>> from sklearn.ensemble import (
        ...     HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        ... )
        >>> adapter = DoubleMLAdapter(
        ...     DoubleMLIRM,
        ...     alpha=0.10,
        ...     ml_g=HistGradientBoostingRegressor(max_iter=200),
        ...     ml_m=HistGradientBoostingClassifier(max_iter=200),
        ... )
    """

    def __init__(
        self,
        model_class: type,
        name: str | None = None,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self._model_class = model_class
        self._kwargs = kwargs
        self._name = name or model_class.__name__
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.supported_outcome_types = self._infer_supported_types(model_class)
        self._fitted_model = None
        self._is_fitted = False

    @staticmethod
    def _infer_supported_types(model_class: type) -> tuple[str, ...]:
        """DoubleMLIRM accepts a classifier ``ml_g`` for binary outcomes;
        other DoubleML classes in our scope are continuous-only."""
        if getattr(model_class, "__name__", "") == "DoubleMLIRM":
            return ("continuous", "binary")
        return ("continuous",)

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return False

    def validate_outcome_type(self, detected: str) -> None:
        """For binary outcomes, the IRM ``ml_g`` learner must implement
        ``predict_proba``. DoubleMLIRM also requires
        ``DoubleMLData(binary_outcome=True)``, which the adapter sets
        automatically at fit time when the detected outcome is binary.
        """
        if detected != "binary":
            return
        ml_g = self._kwargs.get("ml_g")
        if ml_g is not None and not hasattr(ml_g, "predict_proba"):
            raise ValueError(
                f"{self._name}: with a binary outcome, ml_g must be a "
                f"classifier (must implement predict_proba). Got "
                f"{type(ml_g).__name__!r}."
            )

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        **kwargs,
    ) -> None:
        from doubleml import DoubleMLData
        from sklearn.base import clone

        random_state = kwargs.pop("random_state", None)

        # Build DataFrame for DoubleML
        n_features = X.shape[1]
        x_cols = [f"X{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=x_cols)
        df["T"] = T
        df["Y"] = Y

        data = DoubleMLData(df, y_col="Y", d_cols="T", x_cols=x_cols)

        # Clone nuisance models so each call gets fresh estimators
        if random_state is not None:
            sub_rng = np.random.default_rng(random_state)
        init_kwargs = {}
        for key, val in self._kwargs.items():
            if hasattr(val, "fit"):
                cloned = clone(val)
                if random_state is not None and hasattr(cloned, "random_state"):
                    cloned.random_state = int(sub_rng.integers(0, 2**32 - 1))
                init_kwargs[key] = cloned
            else:
                init_kwargs[key] = val

        # DoubleML's sample splitting uses numpy's global random state and
        # exposes no seed parameter. Snapshot np.random's state, seed it
        # deterministically for the duration of DoubleML's construction
        # and fit, then restore — so our fold-level reproducibility
        # doesn't leak into the caller's np.random stream. (See #16.)
        saved_np_state = np.random.get_state() if random_state is not None else None
        try:
            if random_state is not None:
                np.random.seed(int(random_state))
            model = self._model_class(data, **init_kwargs)
            model.fit()
        finally:
            if saved_np_state is not None:
                np.random.set_state(saved_np_state)

        self._fitted_model = model
        self._is_fitted = True

    def ate(self, X: np.ndarray | None = None) -> ComponentAteEstimate:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self._name} is not fitted. Call fit() first."
            )

        model = self._fitted_model

        # DoubleML >= 0.11 uses .coef; older versions use .coef_
        if hasattr(model, "coef"):
            ate = float(model.coef[0])
        else:
            ate = float(model.coef_[0])

        # Extract CI
        ci_lower, ci_upper = None, None
        try:
            ci = model.confint(level=1.0 - self.alpha)
            ci_lower = float(ci.iloc[0, 0])
            ci_upper = float(ci.iloc[0, 1])
        except Exception:
            pass

        return ComponentAteEstimate(
            ate=ate, ci_lower=ci_lower, ci_upper=ci_upper
        )

    def cate(self, X: np.ndarray) -> ComponentCateEstimate:
        raise NotImplementedError(
            f"{self._name} does not support CATE estimation."
        )
