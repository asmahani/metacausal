"""Default method configurations.

This module imports heavy dependencies (econml, doubleml, causalml,
stochtree). It is only loaded when the user calls ``CausalEnsemble()``
without specifying methods, or calls ``default_methods()`` directly.
"""

from __future__ import annotations


def default_propensity_model():
    """Default propensity model: HistGradientBoostingClassifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=10,
    )


def default_outcome_model(outcome_type: str = "continuous"):
    """Default outcome model for the given outcome type.

    Returns ``HistGradientBoostingRegressor`` for continuous outcomes and
    ``HistGradientBoostingClassifier`` for binary outcomes — both with
    early stopping and matching hyperparameters so the only difference is
    the loss/output space.
    """
    if outcome_type == "binary":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=200, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=10,
        )
    if outcome_type == "continuous":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=200, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=10,
        )
    raise ValueError(
        f"Unknown outcome_type: {outcome_type!r}. "
        f"Expected 'continuous' or 'binary'."
    )


def default_methods(outcome_type: str = "continuous") -> list:
    """Return a default set of causal estimators for the given outcome type.

    Requires: econml, doubleml, causalml, stochtree, sklearn.
    Install via ``pip install metacausal[all]`` for the full default library.

    For ``outcome_type="continuous"`` (default) this returns nine components
    spanning four supported frameworks::

        DoubleML:   DoubleMLIRM, DoubleMLPLR
        EconML:     CausalForestDML, DRLearner, TLearner, XLearner
        CausalML:   BaseRRegressor (R-Learner), TMLELearner
        stochtree:  Bayesian Causal Forest

    For ``outcome_type="binary"`` the pool targets the **risk-difference**
    estimand and contains seven components::

        DoubleML:   DoubleMLIRM (classifier ml_g)
        EconML:     CausalForestDML, DRLearner (both with discrete_outcome)
        CausalML:   BaseTClassifier, BaseXClassifier,
                    BaseRClassifier, TMLELearner

    Components dropped on binary:

    - ``DoubleMLPLR``: IRM is the canonical DoubleML choice for binary Y;
      PLR adds an awkward linear-probability fit without diversity.
    - EconML T/X-Learners: no ``discrete_outcome`` flag, ``.predict()``
      would return hard labels. The CausalML ``Base{T,X}Classifier``
      siblings stand in.
    - ``BaseRRegressor``: replaced by ``BaseRClassifier``.
    - ``BaseDRClassifier``: like ``BaseDRRegressor``, needs an externally
      computed propensity passed via ``fit(..., p=...)``; without it,
      predictions are on a degenerate scale. EconML's ``DRLearner`` covers
      the DR family in the binary pool. Users who want it can wrap it
      explicitly with their own propensity wiring.
    - Stochtree BCF: Gaussian outcome model only.
    """
    if outcome_type == "binary":
        return _default_methods_binary()
    if outcome_type != "continuous":
        raise ValueError(
            f"Unknown outcome_type: {outcome_type!r}. "
            f"Expected 'continuous' or 'binary'."
        )

    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    from econml.metalearners import TLearner, XLearner

    from metacausal.adapters.causalml import CausalMLAdapter
    from metacausal.adapters.doubleml import DoubleMLAdapter
    from metacausal.adapters.stochtree import StochtreeAdapter

    # Lazy imports to avoid top-level dependency
    from causalml.inference.meta import BaseRRegressor, TMLELearner
    from doubleml import DoubleMLIRM, DoubleMLPLR

    r_learner = BaseRRegressor(
        outcome_learner=default_outcome_model(),
        propensity_learner=default_propensity_model(),
        effect_learner=default_outcome_model(),
    )
    # cv_n_jobs=1: run the internal cross_val_predict serially. CausalML's
    # BaseRLearner defaults cv_n_jobs=-1 (unlike EconML's n_jobs, which most
    # wrapped estimators default to serial, this knob is on by default), so
    # left alone it nests a joblib pool inside MetaCausal's own outer
    # parallelism (fit/bootstrap) on every fit. Not a constructor kwarg on
    # BaseRRegressor (only exposed on the BaseRLearner parent's __init__),
    # hence the post-construction attribute set. Mirrors the CausalForestDML
    # n_jobs=1 treatment below.
    r_learner.cv_n_jobs = 1
    # model_p=...: work around a CausalML bug (uber/causalml#937, fixed on
    # their master but not yet in a PyPI release as of causalml 0.17.0).
    # BaseRLearner.fit() calls _set_propensity_models(), which reads
    # self.model_p to decide whether to use our HGB propensity_learner or
    # fall back to CausalML's own ElasticNetPropensityModel default -- but
    # self.model_p is only assigned from self.propensity_learner *after*
    # that call, so on a fresh instance our propensity_learner is silently
    # ignored on every fit. ElasticNetPropensityModel's saga+elasticnet grid
    # search then floods ConvergenceWarnings on Lalonde-sized data. Setting
    # model_p directly sidesteps the ordering bug. Harmless once CausalML
    # ships the upstream fix (it would just set the same value twice).
    r_learner.model_p = default_propensity_model()

    return [
        # Semiparametric doubly-robust
        DoubleMLAdapter(DoubleMLIRM, ml_g=default_outcome_model(), ml_m=default_propensity_model()),
        DoubleMLAdapter(DoubleMLPLR, ml_l=default_outcome_model(), ml_m=default_propensity_model()),
        # Tree-based
        CausalForestDML(
            model_y=default_outcome_model(),
            model_t=default_propensity_model(),
            discrete_treatment=True,
            n_estimators=200,
            # n_jobs=1: build the forest serially. EconML defaults to
            # n_jobs=-1, whose internal joblib pool nests inside
            # MetaCausal's own outer parallelism (fit/bootstrap), causing
            # oversubscription and intermittent segfaults in EconML's
            # Cython tree splitter. This mirrors the n_jobs=1 guidance for
            # user-supplied components in the parallel-execution section.
            n_jobs=1,
        ),
        # DR-Learner
        DRLearner(model_regression=default_outcome_model(), model_propensity=default_propensity_model()),
        # Meta-learners
        TLearner(models=default_outcome_model()),
        XLearner(
            models=default_outcome_model(),
            propensity_model=default_propensity_model(),
        ),
        # R-Learner (CausalML — auto-wrapped by CausalEnsemble)
        r_learner,
        # TMLE (CausalML, ATE-only — explicit adapter so the propensity
        # model is fitted via CausalMLAdapter's dedicated slot)
        CausalMLAdapter(
            TMLELearner(learner=default_outcome_model()),
            propensity_model=default_propensity_model(),
        ),
        # Bayesian Causal Forest (stochtree — explicit adapter required)
        StochtreeAdapter(propensity_model=default_propensity_model()),
    ]


def _default_methods_binary() -> list:
    """Build the seven-component binary-outcome default pool.

    Effect-side learners (``effect_learner``, ``treatment_effect_learner``,
    TMLE's outcome learner) stay regressors even on binary Y: the
    treatment effect itself is a continuous risk-difference, and TMLE
    applies its own logit transform internally.
    """
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner

    from metacausal.adapters.causalml import CausalMLAdapter
    from metacausal.adapters.doubleml import DoubleMLAdapter

    from causalml.inference.meta import (
        BaseTClassifier,
        BaseXClassifier,
        BaseRClassifier,
        TMLELearner,
    )
    from doubleml import DoubleMLIRM

    out_clf = lambda: default_outcome_model("binary")
    out_reg = lambda: default_outcome_model("continuous")
    prop = default_propensity_model

    r_classifier = BaseRClassifier(
        outcome_learner=out_clf(),
        effect_learner=out_reg(),
        propensity_learner=prop(),
    )
    # cv_n_jobs=1: see the continuous pool's BaseRRegressor comment above.
    r_classifier.cv_n_jobs = 1
    # model_p=...: see the continuous pool's BaseRRegressor comment above
    # (uber/causalml#937) -- BaseRClassifier shares the same bug.
    r_classifier.model_p = prop()

    return [
        # Semiparametric doubly-robust
        DoubleMLAdapter(DoubleMLIRM, ml_g=out_clf(), ml_m=prop()),
        # Tree-based
        CausalForestDML(
            model_y=out_clf(),
            model_t=prop(),
            discrete_outcome=True,
            discrete_treatment=True,
            n_estimators=200,
            n_jobs=1,  # serial forest; see continuous pool for rationale
        ),
        # DR-Learner
        DRLearner(
            model_regression=out_clf(),
            model_propensity=prop(),
            discrete_outcome=True,
        ),
        # Meta-learner classifiers (auto-wrapped by CausalEnsemble)
        BaseTClassifier(learner=out_clf()),
        # X-Learner classifier: wrapped explicitly so the adapter can
        # pre-fit our HGB propensity and pass it through ``fit(..., p=...)``,
        # avoiding CausalML's internal ElasticNetPropensityModel default
        # (LogisticRegressionCV with saga, which floods ConvergenceWarnings).
        CausalMLAdapter(
            BaseXClassifier(outcome_learner=out_clf(), effect_learner=out_reg()),
            propensity_model=prop(),
        ),
        r_classifier,
        # TMLE — outcome learner stays a regressor (internal logit)
        CausalMLAdapter(
            TMLELearner(learner=out_reg()),
            propensity_model=prop(),
        ),
    ]
