"""Warning classes for metacausal-specific concerns.

Hierarchy::

    Warning
    └── MetaCausalWarning            (umbrella for all metacausal warnings)
        ├── ComponentWarning         (operating-pool concerns)
        │   ├── ComponentFailureWarning   (component fit/predict failures)
        │   └── ComponentExclusionWarning (configuration mismatches)
        └── BootstrapWarning         (bootstrap inference reliability)

The classes are re-exported from :mod:`metacausal`; import as
``from metacausal import MetaCausalWarning, ComponentWarning,
ComponentFailureWarning, ComponentExclusionWarning, BootstrapWarning``.

The hierarchy lets users filter at the appropriate granularity::

    import warnings
    import metacausal as mc

    # Catch any metacausal-emitted warning (most strict)
    warnings.filterwarnings("error", category=mc.MetaCausalWarning)

    # Catch only operating-pool concerns (failure or exclusion)
    warnings.filterwarnings("error", category=mc.ComponentWarning)

    # Catch only fit/predict-time runtime failures
    warnings.filterwarnings("error", category=mc.ComponentFailureWarning)

    # Catch only configuration mismatches (outcome-type, ATE-only)
    warnings.filterwarnings("error", category=mc.ComponentExclusionWarning)

    # Catch only bootstrap inference reliability concerns
    warnings.filterwarnings("error", category=mc.BootstrapWarning)

Classes inherit from :class:`Warning` directly (not :class:`RuntimeWarning`)
so that metacausal-scoped policies are independent of broad
:class:`RuntimeWarning` behaviour applied by the user or other libraries
(numpy, scipy, sklearn).
"""

from __future__ import annotations


class MetaCausalWarning(Warning):
    """Umbrella class for all metacausal-emitted warnings.

    Currently subclassed into :class:`ComponentWarning` (operating-pool
    concerns: fit failures and configuration exclusions) and
    :class:`BootstrapWarning` (bootstrap inference reliability concerns).
    Filtering at this level catches any metacausal-emitted warning
    regardless of subcategory.
    """


class ComponentWarning(MetaCausalWarning):
    """Base class for warnings indicating an operating-ensemble pool
    reduction.

    Subclassed into :class:`ComponentFailureWarning` (runtime fit/predict
    failures) and :class:`ComponentExclusionWarning` (configuration
    mismatches). Filtering at this level catches any partial-pool surprise
    regardless of cause, but does NOT catch :class:`BootstrapWarning`
    (which is about inference reliability, not pool composition).
    """


class ComponentFailureWarning(ComponentWarning):
    """A component raised an exception during fit, cross-fitting,
    prediction, or as part of nuisance/weight-fitting machinery, and
    was silently excluded from the operating ensemble.

    Data-dependent: the same configuration on different data may or may
    not trigger this warning.
    """


class ComponentExclusionWarning(ComponentWarning):
    """A component was excluded by configuration mismatch.

    Two cases:

    - The component's declared ``supported_outcome_types`` did not
      include the detected outcome type at fit time.
    - The component is ATE-only (``supports_cate=False``) and the
      active aggregation strategy requires CATE-capable adapters
      (i.e., :class:`SupervisedStrategy`).

    Deterministic given the user's configuration and the detected
    outcome type.
    """


class BootstrapWarning(MetaCausalWarning):
    """A bootstrap inference call hit a reliability concern.

    Currently emitted when the bootstrap replicate failure rate exceeds
    10% (CIs may be unreliable) or when all replicates failed (CIs are
    not computed). A bootstrap-replicate failure can arise from
    component issues *within* a replicate, but also from global causes
    (numerical issues on the resample, weight-optimisation failure,
    etc.), so this category is distinct from :class:`ComponentWarning`
    — it is about *inference reliability*, not pool composition.
    """
