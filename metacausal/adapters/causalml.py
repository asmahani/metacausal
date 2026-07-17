"""Adapter for CausalML estimators (Uber, causalml>=0.14)."""

from __future__ import annotations

import copy
import inspect
import os
from typing import Any

import numpy as np

from metacausal._parallel import INNER_WORKER_ENV, force_serial
from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

_CAUSALML_MODULE_PREFIX = "causalml.inference"
_TMLE_CLASS_NAME = "TMLELearner"


def _is_causalml(obj) -> bool:
    """Return True if obj is a CausalML estimator (lazy — no causalml import)."""
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith(_CAUSALML_MODULE_PREFIX)


def _is_tmle(obj) -> bool:
    return type(obj).__name__ == _TMLE_CLASS_NAME


def _is_uplift(obj) -> bool:
    """Return True for UpliftTree/Forest classes that require string treatment."""
    module = getattr(type(obj), "__module__", "") or ""
    return "uplift" in module


# ---------------------------------------------------------------------------
# CausalMLAdapter
# ---------------------------------------------------------------------------


class CausalMLAdapter:
    """Wrap a CausalML estimator as a metacausal causal estimator.

    Supports all three estimation families from the ``causalml`` package:

    **Meta-learners** (``causalml.inference.meta``)
        BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseDRRegressor,
        BaseRRegressor, and their Classifier variants.
        ``predict(X)`` returns ``(n, 1)``; squeezed to ``(n,)``.
        ``estimate_ate()`` returns ``(ate, lb, ub)`` arrays of shape ``(1,)``.

    **Tree-based methods** (``causalml.inference.tree``)
        CausalTreeRegressor: ``predict(X)`` returns ``(n,)`` directly.
        UpliftTreeClassifier, UpliftRandomForestClassifier: ``predict(X)``
        returns ``(n, 2)`` — P(Y=1|control) and P(Y=1|treatment). Uplift
        (CATE) is computed as treatment column minus control column. These
        models require string treatment labels; the adapter converts binary
        ``T`` automatically using ``model.control_name`` for control and
        ``"treatment"`` for the treated group.

    **TMLE** (``TMLELearner`` from ``causalml.inference.meta``)
        ATE-only (``supports_cate=False``). ``TMLELearner`` has no ``fit()``
        or ``predict()`` — it does all work in ``estimate_ate(X, T, Y, p)``
        where ``p`` is a required propensity-score array. The adapter fits
        the propensity model during ``fit()``, seeds TMLE's internal
        outcome learner, runs ``estimate_ate`` once, and caches the
        ``(ate, lb, ub)`` triple; ``ate()`` is then a constant-time lookup.
        Caching is necessary because TMLE's ``estimate_ate`` refits its
        own outcome learner (with an un-seeded validation split) per call,
        which makes successive calls with identical inputs drift. Provide
        a ``propensity_model`` at construction time to customise the
        propensity estimator.

    Parameters
    ----------
    model : CausalML estimator
        An initialized (unfitted) CausalML estimator instance.
    name : str or None
        Display name. Defaults to the class name of the wrapped model.
    propensity_model : sklearn classifier or None
        Used only for ``TMLELearner``. Fitted during ``fit()`` to produce
        the propensity scores ``p`` that TMLE requires. Default:
        ``LogisticRegression(max_iter=500)``. Ignored for all other
        CausalML estimators (they handle propensity internally or accept it
        optionally via their own ``p`` parameter).

    Notes
    -----
    **Seed / reproducibility:** CausalML has no uniform ``random_state``
    in ``fit()``. The adapter passes ``random_state`` as ``seed`` for
    ``BaseDRRegressor`` and ``BaseRRegressor`` (the only classes that accept
    it). For all other classes, set ``random_state`` at model construction
    time. CausalEnsemble forwards seeds but the wrapped model may ignore them.

    **Deep-copy requirement:** the wrapped model must be deep-copyable for
    bootstrap resampling and cross-fitting. Standard sklearn-backed CausalML
    models are deep-copyable; models with GPU tensors or open file handles
    may fail.

    Examples
    --------
    >>> from causalml.inference.meta import BaseTRegressor
    >>> from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    >>> from metacausal.adapters import CausalMLAdapter
    >>> from metacausal.datasets import load_lalonde
    >>> X, T, Y = load_lalonde()
    >>> t_learner = CausalMLAdapter(BaseTRegressor(learner=HGBR(max_iter=20)))
    >>> t_learner.fit(X, T, Y, random_state=42)
    >>> result = t_learner.ate()
    >>> result
    ComponentAteEstimate(ate=..., ci=[..., ...])
    """

    def __init__(
        self,
        model: Any,
        name: str | None = None,
        propensity_model: Any = None,
    ) -> None:
        self._model = model
        self._name = name or type(model).__name__
        self._propensity_model = propensity_model
        self.supported_outcome_types = self._infer_supported_types(model)
        self._fitted_model = None
        self._X_train: np.ndarray | None = None
        self._T_train: np.ndarray | None = None   # as passed to fit() (may be strings)
        self._Y_train: np.ndarray | None = None
        self._p_hat: np.ndarray | None = None     # propensity scores (TMLE only)
        # Fitted propensity model — populated only when the non-TMLE branch
        # of fit() pre-fits ``self._propensity_model`` to thread ``p=``
        # through both fit and predict on the wrapped model.
        self._fitted_propensity_model: Any | None = None
        # Cached (ate, lb, ub) triple for TMLE — populated in _fit_tmle.
        # TMLE's estimate_ate is non-deterministic (refits its internal
        # outcome learner with an un-seeded val split). We run it once at
        # fit time and return the cached value from ate() thereafter. (#16)
        self._tmle_ate_cache: tuple[float, float, float] | None = None
        self._is_fitted = False
        # Column indices for uplift tree predict() output (set during fit)
        self._control_col: int | None = None
        self._treatment_col: int | None = None

    @staticmethod
    def _infer_supported_types(model: Any) -> tuple[str, ...]:
        """Map CausalML class to its outcome-type capability.

        - ``TMLELearner`` accepts both: it scales Y to [0,1] and applies
          its own logit transform internally, so a regressor in the
          outcome slot works for either type.
        - Meta-learner ``*Classifier`` siblings (``BaseRClassifier`` etc.)
          and uplift tree/forest are binary-only by construction.
        - Everything else (``*Regressor``, ``CausalTreeRegressor``) is
          continuous-only.
        """
        cls_name = type(model).__name__
        if cls_name == _TMLE_CLASS_NAME:
            return ("continuous", "binary")
        if cls_name.endswith("Classifier"):
            return ("binary",)
        return ("continuous",)

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_cate(self) -> bool:
        return not _is_tmle(self._model)

    def validate_outcome_type(self, detected: str) -> None:
        """Validate the wrapped CausalML model's nuisance configuration.

        ``TMLELearner`` is the one quirk worth catching: even for binary
        outcomes, its ``learner`` slot must be a regressor — TMLE's
        internal logit transform expects raw conditional means, not
        probabilities. For the meta-learner ``*Classifier`` family,
        outcome-slot classifier-ness is enforced by CausalML itself; the
        capability declaration here keeps them out of the continuous
        path so the misuse can't arise.
        """
        if _is_tmle(self._model):
            learner = getattr(self._model, "learner", None)
            if learner is not None and hasattr(learner, "predict_proba"):
                raise ValueError(
                    f"{self._name}: TMLELearner expects a regressor in its "
                    f"'learner' slot (it applies its own logit transform "
                    f"internally). Got a classifier-like learner of type "
                    f"{type(learner).__name__!r}."
                )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit the wrapped CausalML model.

        For ``TMLELearner``: fits the propensity model on ``(X, T)`` and
        caches the scores. TMLE itself has no ``fit()`` — all its work
        happens in ``ate()`` via ``estimate_ate()``.

        For all other classes: deep-copies the model, maps our
        ``(X, T, Y)`` convention to CausalML's ``(X, treatment, y)``
        convention, and calls ``model.fit()``. Passes ``random_state`` as
        ``seed`` for estimators whose ``fit()`` accepts that parameter
        (e.g. ``BaseDRRegressor``).
        """
        random_state = kwargs.pop("random_state", None)
        model = copy.deepcopy(self._model)

        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)

        self._X_train = X
        self._Y_train = Y
        self._is_fitted = True
        self._fitted_model = model

        if _is_tmle(model):
            self._fit_tmle(X, T, Y, random_state)
            return

        if random_state is not None:
            self._seed_model(model, random_state)

        if os.environ.get(INNER_WORKER_ENV):
            self._force_serial(model)

        # Seed injection (where fit() accepts it)
        fit_kwargs: dict = {}
        fit_sig = inspect.signature(model.fit)
        if random_state is not None and "seed" in fit_sig.parameters:
            fit_kwargs["seed"] = int(random_state)

        # Propensity injection: when the wrapped CausalML estimator accepts
        # a precomputed propensity (``fit(..., p=...)``) and the user wired
        # a ``propensity_model``, pre-fit it and pass the scores through.
        # Without this, BaseXLearner/Classifier silently constructs
        # ``ElasticNetPropensityModel()`` (LogisticRegressionCV with saga),
        # which floods stderr with ConvergenceWarnings on small data.
        # The fitted propensity model is also stashed for predict-time
        # use in cate(): when fit receives ``p=``, BaseXLearner skips
        # populating its own ``self.propensity_model`` attribute, so
        # predict on new X needs the same propensity threading.
        if self._propensity_model is not None and "p" in fit_sig.parameters:
            prop = copy.deepcopy(self._propensity_model)
            if random_state is not None and hasattr(prop, "random_state"):
                prop.random_state = int(random_state)
            prop.fit(X, T)
            fit_kwargs["p"] = prop.predict_proba(X)[:, 1]
            self._fitted_propensity_model = prop

        # Uplift tree/forest: convert binary T to string treatment labels
        T_fit = self._encode_treatment(model, T)
        self._T_train = T_fit

        model.fit(X, T_fit, Y, **fit_kwargs)

        # For uplift tree/forest: resolve column indices from classes_
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            ctrl_name = model.control_name
            self._control_col = classes.index(ctrl_name)
            non_ctrl = [i for i, c in enumerate(classes) if str(c) != str(ctrl_name)]
            self._treatment_col = non_ctrl[0] if non_ctrl else 1 - self._control_col

    def _fit_tmle(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        random_state: int | None,
    ) -> None:
        """Fit the propensity model, run TMLE once, and cache the ATE.

        TMLE has no ``fit()`` of its own — all its work happens in
        ``estimate_ate(X, T, Y, p)``. That method lazily fits its
        internal outcome learner (``model_tau``) each call and does not
        seed it, so repeated calls with identical inputs drift. We run
        ``estimate_ate`` exactly once here — with a deterministically
        seeded ``model_tau`` — and cache the ``(ate, lb, ub)`` triple,
        so ``adapter.ate()`` is a constant-time cached lookup. (See #16.)
        """
        from sklearn.linear_model import LogisticRegression

        prop = copy.deepcopy(
            self._propensity_model or LogisticRegression(max_iter=500)
        )
        if random_state is not None and hasattr(prop, "random_state"):
            prop.random_state = int(random_state)

        prop.fit(X, T)
        self._p_hat = prop.predict_proba(X)[:, 1]
        self._T_train = T   # binary integers, as TMLE expects

        # Seed TMLE's internal outcome learner so estimate_ate is
        # reproducible across processes given the same random_state.
        if random_state is not None:
            rng = np.random.default_rng(random_state)
            tau = getattr(self._fitted_model, "model_tau", None)
            if tau is not None and hasattr(tau, "random_state"):
                tau.random_state = int(rng.integers(0, 2**31 - 1))

        # Run TMLE once, cache the triple.
        result = self._fitted_model.estimate_ate(X, T, Y, p=self._p_hat)
        self._tmle_ate_cache = (
            float(np.asarray(result[0]).ravel()[0]),
            float(np.asarray(result[1]).ravel()[0]),
            float(np.asarray(result[2]).ravel()[0]),
        )

    # ------------------------------------------------------------------
    # cate
    # ------------------------------------------------------------------

    def cate(self, X: np.ndarray) -> ComponentCateEstimate:
        """Return CATE predictions.

        Squeezes ``(n, 1)`` meta-learner output to ``(n,)``. For uplift
        tree/forest, computes ``P(Y=1|T=1,X) - P(Y=1|T=0,X)``.
        Raises ``NotImplementedError`` for ``TMLELearner`` (ATE-only).
        """
        if not self._is_fitted:
            raise RuntimeError(f"'{self._name}' is not fitted. Call fit() first.")
        if _is_tmle(self._fitted_model):
            raise NotImplementedError(
                f"'{self._name}' (TMLELearner) is an ATE-only estimator and "
                "does not support per-unit CATE predictions."
            )

        predict_kwargs: dict = {}
        sig = inspect.signature(self._fitted_model.predict)
        if "verbose" in sig.parameters:
            predict_kwargs["verbose"] = False

        # Thread propensity into predict for X-Learner-style estimators.
        # When fit() received ``p=``, the wrapped model did not populate
        # its own ``self.propensity_model``, so the predict-time fallback
        # would crash on new X. Use the propensity model the adapter
        # stored at fit time to compute scores on the evaluation X.
        if self._fitted_propensity_model is not None and "p" in sig.parameters:
            predict_kwargs["p"] = self._fitted_propensity_model.predict_proba(
                np.asarray(X)
            )[:, 1]

        raw = np.asarray(
            self._fitted_model.predict(np.asarray(X), **predict_kwargs),
            dtype=float,
        )

        if raw.ndim == 2:
            if self._treatment_col is not None and raw.shape[1] >= 2:
                # UpliftTreeClassifier (n, 2): uplift = treatment_prob - control_prob
                cate = raw[:, self._treatment_col] - raw[:, self._control_col]
            else:
                # Meta-learner (n, 1) or UpliftRandomForestClassifier (n, 1):
                # column 0 is already the effect of interest
                cate = raw[:, 0]
        else:
            # CausalTreeRegressor: already (n,)
            cate = raw

        return ComponentCateEstimate(cate=cate)

    # ------------------------------------------------------------------
    # ate
    # ------------------------------------------------------------------

    def ate(self, X: np.ndarray | None = None) -> ComponentAteEstimate:
        """Return the ATE.

        **TMLELearner:** calls ``estimate_ate(X_train, T_train, Y_train,
        p=p_hat)`` using training data and cached propensity scores. The
        ``X`` argument is ignored — TMLE's targeted update is tied to the
        training sample.

        **All other estimators:** if ``estimate_ate()`` is available, calls
        it on the training data (regardless of ``X``). CausalML's DR-based
        ATE estimators require the original outcome and treatment data.
        Falls back to ``mean(cate(X_eval))`` for estimators without
        ``estimate_ate()`` (e.g. uplift tree/forest).
        """
        if not self._is_fitted:
            raise RuntimeError(f"'{self._name}' is not fitted. Call fit() first.")

        X_eval = np.asarray(X) if X is not None else self._X_train

        if _is_tmle(self._fitted_model):
            # TMLE's ATE was computed and cached in _fit_tmle — see the
            # reasoning there. Return the cached triple unchanged.
            assert self._tmle_ate_cache is not None
            ate, lb, ub = self._tmle_ate_cache
            return ComponentAteEstimate(ate=ate, ci_lower=lb, ci_upper=ub)

        if hasattr(self._fitted_model, "estimate_ate"):
            # pretrain=True when supported: avoids silent refitting on every
            # call. The default (pretrain=False) re-runs fit_predict() inside
            # estimate_ate(), which (a) is expensive and (b) makes successive
            # calls non-deterministic because internal cross-fitting isn't
            # fully seeded. With pretrain=True the ATE is the mean of the
            # stored model's predictions — consistent with how cate() uses
            # the same stored fit — and the native analytical CI is still
            # returned. (See #16.)
            try:
                ate_kwargs: dict = {}
                sig = inspect.signature(self._fitted_model.estimate_ate)
                if "pretrain" in sig.parameters:
                    ate_kwargs["pretrain"] = True
                result = self._fitted_model.estimate_ate(
                    self._X_train, self._T_train, self._Y_train, **ate_kwargs,
                )
                ate_val = float(np.asarray(result[0]).ravel()[0])
                lb = float(np.asarray(result[1]).ravel()[0])
                ub = float(np.asarray(result[2]).ravel()[0])
                return ComponentAteEstimate(ate=ate_val, ci_lower=lb, ci_upper=ub)
            except Exception:
                pass  # fall through to cate-based fallback

        # Fallback: mean CATE (uplift tree/forest, or estimate_ate failure)
        return ComponentAteEstimate(ate=float(self.cate(X_eval).cate.mean()))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_model(model: Any, random_state: int) -> None:
        """Propagate ``random_state`` into *model* and every nested
        stateful attribute.

        Walks ``vars(model)`` (instance ``__dict__``, insertion-ordered)
        and sets ``random_state`` on each attribute that exposes one —
        nuisance learners, effect learners, propensity learners, CV
        splitters. Each gets an independent seed drawn sequentially from
        a single RNG deterministically derived from ``random_state``,
        so two fresh deep-copies of the same template seeded with the
        same ``random_state`` end up in matching states.

        Introspection is necessary because causalml stores learners
        under different internal names per class (``model``,
        ``model_c``/``model_t``, ``model_mu``/``model_tau``/``model_p``,
        ``model_mu_c``/``model_mu_t``/``model_tau``, …) — and also
        holds CV splitters like ``KFold`` that cache the seed at
        construction and ignore later writes to ``model.random_state``.
        A hardcoded attribute list (as used previously) silently missed
        every one of these. See #16.
        """
        rng = np.random.default_rng(random_state)
        if hasattr(model, "random_state"):
            model.random_state = int(rng.integers(0, 2**31 - 1))
        for attr_value in vars(model).values():
            if isinstance(attr_value, list):
                for item in attr_value:
                    if hasattr(item, "random_state"):
                        item.random_state = int(rng.integers(0, 2**31 - 1))
            elif hasattr(attr_value, "random_state"):
                attr_value.random_state = int(rng.integers(0, 2**31 - 1))

    @staticmethod
    def _force_serial(model: Any) -> None:
        """Pin CausalML's ``cv_n_jobs`` to 1 when fitting inside a
        MetaCausal worker (see
        :data:`~metacausal._parallel.INNER_WORKER_ENV`).

        ``BaseRLearner`` (the R-Learner family: ``BaseRRegressor``,
        ``BaseRClassifier``) defaults ``cv_n_jobs=-1`` for its internal
        ``cross_val_predict`` call. Unlike EconML's ``n_jobs`` -- which most
        wrapped estimators default to serial -- this knob is *on* by
        default, so left alone it nests a second joblib pool inside the
        outer worker on every single fit, not just when a user explicitly
        opts in. No ``sub_attrs`` list is passed to
        :func:`~metacausal._parallel.force_serial`: ``cv_n_jobs`` lives on
        the top-level R-learner object itself, not on a nested per-fold
        learner slot (``model_mu``/``model_tau`` are plain learners with no
        cross-validation of their own), so the generic ``vars(model)`` walk
        is a no-op beyond the top-level pin for every class in the current
        pool -- and stays correct automatically if a future CausalML class
        nests another ``cv_n_jobs``-bearing object.  Outside a worker (the
        serial outer path), the model keeps its own setting.
        """
        force_serial(model, "cv_n_jobs")

    @staticmethod
    def _encode_treatment(model: Any, T: np.ndarray) -> np.ndarray:
        """Convert binary T (0/1) to string labels for uplift tree/forest."""
        if _is_uplift(model):
            ctrl = str(model.control_name)
            return np.where(T == 0, ctrl, "treatment")
        return T
