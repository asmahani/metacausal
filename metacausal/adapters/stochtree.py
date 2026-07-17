"""Adapter for stochtree BCF (Bayesian Causal Forest)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


class StochtreeAdapter:
    """Wrap stochtree's BCFModel for use in CausalEnsemble.

    BCF requires propensity scores as input. By default, this adapter
    computes cross-fitted propensity scores using the provided
    ``propensity_model``. A pre-computed propensity array can be
    passed via ``fit(..., propensity=ps)`` to skip this step.

    Parameters:
        name: Display name (default ``"BCF"``).
        propensity_model: An sklearn-compatible classifier for propensity
            estimation. Must support ``fit(X, T)`` and ``predict_proba(X)``.
            If ``None``, uses ``HistGradientBoostingClassifier`` with
            early stopping.
        num_gfr: Number of "grow-from-root" warm-start iterations
            (default 5). Set to 0 to disable.
        num_burnin: MCMC burn-in iterations (default 200).
        num_mcmc: Posterior samples after burn-in (default 200).
        alpha: Credible interval level (default 0.05 for 95% CI).
        propensity_n_splits: CV folds for propensity estimation (default 5).
        propensity_clip_eps: Clip propensities to
            ``[eps, 1 - eps]`` (default 0.01).
        general_params: Optional dict passed to
            ``BCFModel.sample(general_params=...)``. See stochtree docs for
            keys such as ``"random_seed"``, ``"propensity_covariate"``,
            ``"adaptive_coding"``, ``"standardize"``, ``"num_chains"``, etc.
        prognostic_forest_params: Optional dict passed to
            ``BCFModel.sample(prognostic_forest_params=...)``. Common keys:
            ``"num_trees"`` (default 250), ``"alpha"``, ``"beta"``,
            ``"min_samples_leaf"``, ``"max_depth"``.
        treatment_effect_forest_params: Optional dict passed to
            ``BCFModel.sample(treatment_effect_forest_params=...)``. Common
            keys: ``"num_trees"`` (default 50), ``"alpha"``, ``"beta"``,
            ``"min_samples_leaf"``, ``"max_depth"``.

    Examples:
        Small ``num_gfr``/``num_burnin``/``num_mcmc`` below keep this a
        fast example; use more posterior draws in practice.

        >>> from metacausal.adapters.stochtree import StochtreeAdapter
        >>> from metacausal.datasets import load_lalonde
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, T, Y = load_lalonde()
        >>> bcf = StochtreeAdapter(
        ...     propensity_model=LogisticRegression(max_iter=2000),
        ...     num_gfr=2,
        ...     num_burnin=20,
        ...     num_mcmc=20,
        ... )
        >>> bcf.fit(X, T, Y, random_state=42)
        >>> result = bcf.ate()
        >>> result
        ComponentAteEstimate(ate=..., ci=[..., ...])
    """

    def __init__(
        self,
        name: str = "BCF",
        *,
        propensity_model: Any = None,
        num_gfr: int = 5,
        num_burnin: int = 200,
        num_mcmc: int = 200,
        alpha: float = 0.05,
        propensity_n_splits: int = 5,
        propensity_clip_eps: float = 0.01,
        general_params: dict | None = None,
        prognostic_forest_params: dict | None = None,
        treatment_effect_forest_params: dict | None = None,
    ) -> None:
        self._name = name
        self._propensity_model = propensity_model
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.alpha = alpha
        self.propensity_n_splits = propensity_n_splits
        self.propensity_clip_eps = propensity_clip_eps
        self.general_params = general_params
        self.prognostic_forest_params = prognostic_forest_params
        self.treatment_effect_forest_params = treatment_effect_forest_params

        self._fitted_model = None
        self._tau_samples: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self._T_train: np.ndarray | None = None
        self._propensity_train: np.ndarray | None = None
        self._fitted_propensity_models: list | None = None
        self._is_fitted = False

    supported_outcome_types: tuple[str, ...] = ("continuous",)

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return True

    def validate_outcome_type(self, detected: str) -> None:
        """No-op: BCF has no user-configurable outcome learner. Routing
        away from binary outcomes happens via ``supported_outcome_types``.
        """
        return

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        **kwargs: Any,
    ) -> None:
        from stochtree import BCFModel

        random_state = kwargs.pop("random_state", None)
        propensity = kwargs.pop("propensity", None)

        # Build general_params, injecting ensemble seed if provided
        gp = dict(self.general_params or {})
        if random_state is not None:
            # stochtree's C++ RNG requires a signed 32-bit int
            gp["random_seed"] = int(random_state) % (2**31)

        # Derive a separate seed for propensity CV to avoid sharing
        # the same random stream as the BCF sampler
        bcf_seed = gp.get("random_seed")
        if bcf_seed is not None:
            propensity_seed = (bcf_seed + 1) % (2**31)
        else:
            propensity_seed = None

        # Compute propensity scores if not provided
        if propensity is None:
            propensity = self._compute_propensity(X, T, propensity_seed)

        # Pin BLAS/OpenMP thread pools to 1 during the C++ sampler call.
        # stochtree's native sampler segfaults on macOS when other numeric
        # libraries (EconML, scikit-learn HGB, etc.) have left the OpenMP
        # threadpool in a multi-threaded state earlier in the process.
        from threadpoolctl import threadpool_limits

        model = BCFModel()
        with threadpool_limits(limits=1):
            model.sample(
                X_train=X,
                Z_train=T.astype(float),
                y_train=Y.astype(float),
                propensity_train=propensity,
                num_gfr=self.num_gfr,
                num_burnin=self.num_burnin,
                num_mcmc=self.num_mcmc,
                general_params=gp,
                prognostic_forest_params=self.prognostic_forest_params or {},
                treatment_effect_forest_params=self.treatment_effect_forest_params or {},
            )

        # Posterior draws of treatment effect: shape (n_obs, num_mcmc)
        tau_samples = model.predict(
            X=X,
            Z=T.astype(float),
            propensity=propensity,
            type="posterior",
            terms="cate",
        )

        self._fitted_model = model
        self._tau_samples = tau_samples
        self._X_train = X
        self._T_train = T
        self._propensity_train = propensity
        self._is_fitted = True

    def ate(self, X: np.ndarray | None = None) -> ComponentAteEstimate:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self._name} is not fitted. Call fit() first."
            )

        tau_samples = self._get_tau_samples(X)

        # ATE draws: average over observations for each posterior sample
        # tau_samples shape is (n_obs, num_mcmc)
        ate_draws = tau_samples.mean(axis=0)

        ate = float(ate_draws.mean())
        ci_lower = float(np.quantile(ate_draws, self.alpha / 2))
        ci_upper = float(np.quantile(ate_draws, 1 - self.alpha / 2))

        return ComponentAteEstimate(
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={"ate_draws": ate_draws},
        )

    def _get_tau_samples(self, X: np.ndarray | None) -> np.ndarray:
        """Get tau_samples for the given X, reusing stored samples when possible."""
        if X is None:
            return self._tau_samples
        # Out-of-sample prediction
        propensity = self._predict_propensity(X)
        return self._fitted_model.predict(
            X=X,
            Z=np.ones(X.shape[0]),
            propensity=propensity,
            type="posterior",
            terms="cate",
        )

    def _predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """Estimate propensity scores for new observations.

        Uses the CV-fitted propensity models from ``fit()``, averaging
        their predictions. Falls back to 0.5 if no fitted models are
        available (e.g., propensity was provided manually during fit).
        """
        if self._fitted_propensity_models is None:
            warnings.warn(
                "No fitted propensity models available for out-of-sample "
                "prediction (propensity was provided manually during fit). "
                "Using propensity=0.5 as fallback.",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.full(X.shape[0], 0.5)

        # Average predictions across CV fold models
        proba_sum = np.zeros(X.shape[0])
        for clf in self._fitted_propensity_models:
            proba_sum += clf.predict_proba(X)[:, 1]
        ps = proba_sum / len(self._fitted_propensity_models)

        eps = self.propensity_clip_eps
        return np.clip(ps, eps, 1 - eps)

    def cate(self, X: np.ndarray) -> ComponentCateEstimate:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self._name} is not fitted. Call fit() first."
            )

        tau_samples = self._get_tau_samples(X)

        # CATE point estimates: posterior mean per observation
        # tau_samples shape is (n_obs, num_mcmc)
        cate = tau_samples.mean(axis=1)

        # Credible intervals: quantiles of posterior per observation
        ci_lower = np.quantile(tau_samples, self.alpha / 2, axis=1)
        ci_upper = np.quantile(tau_samples, 1 - self.alpha / 2, axis=1)

        return ComponentCateEstimate(
            cate=cate, ci_lower=ci_lower, ci_upper=ci_upper
        )

    def __getstate__(self) -> dict:
        """Custom pickle support: serialize BCFModel to JSON string.

        ``stochtree_cpp.ForestContainerCpp`` (held inside a fitted
        ``BCFModel``) is a C++ extension type that cannot be pickled by
        joblib's loky backend.  ``BCFModel.to_json()`` produces a complete
        string representation that survives the round-trip exactly.
        """
        state = self.__dict__.copy()
        if state.get("_fitted_model") is not None:
            state["_fitted_model_json"] = state.pop("_fitted_model").to_json()
        return state

    def __setstate__(self, state: dict) -> None:
        """Reconstruct BCFModel from JSON string after unpickling."""
        json_str = state.pop("_fitted_model_json", None)
        self.__dict__.update(state)
        if json_str is not None:
            from stochtree import BCFModel
            model = BCFModel()
            model.from_json(json_str)
            self._fitted_model = model

    def _compute_propensity(
        self,
        X: np.ndarray,
        T: np.ndarray,
        seed: int | None,
    ) -> np.ndarray:
        """Cross-fitted propensity scores."""
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold

        if self._propensity_model is not None:
            base_model = self._propensity_model
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            base_model = HistGradientBoostingClassifier(
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=seed,
            )

        ps = np.full(len(T), np.nan)
        fitted_models = []
        cv = StratifiedKFold(
            n_splits=self.propensity_n_splits, shuffle=True, random_state=seed
        )

        for train_idx, test_idx in cv.split(X, T):
            clf = clone(base_model)
            if hasattr(clf, "random_state"):
                clf.random_state = seed
            clf.fit(X[train_idx], T[train_idx])
            ps[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            fitted_models.append(clf)

        # Store fitted models for out-of-sample propensity prediction
        self._fitted_propensity_models = fitted_models

        eps = self.propensity_clip_eps
        return np.clip(ps, eps, 1 - eps)
