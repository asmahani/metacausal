"""MetaCausal — Robust treatment effect estimation via ensembling."""

import warnings
from importlib.metadata import PackageNotFoundError, version as _version

from metacausal.aggregation import (
    AggregationStrategy,
    AgreementStrategy,
    BootstrapResult,
    CBA,
    EnsembleWeights,
    Mean,
    Median,
    PointwiseStrategy,
    TrimmedMean,
)
from metacausal.adapters.causalml import CausalMLAdapter
from metacausal.adapters.doubleml import DoubleMLAdapter
from metacausal.adapters.econml import EconMLAdapter
from metacausal.adapters.generic import GenericAdapter, GenericATEAdapter, GenericCATEAdapter
from metacausal.adapters.stochtree import StochtreeAdapter
from metacausal.ensemble import CausalEnsemble
from metacausal.estimators import (
    AteEstimate,
    CausalEstimate,
    CateEstimate,
    ComponentAteEstimate,
    ComponentCateEstimate,
    EnsembleEstimate,
)
from metacausal.outcome_type import infer_outcome_type
from metacausal._warnings import (
    BootstrapWarning,
    ComponentExclusionWarning,
    ComponentFailureWarning,
    ComponentWarning,
    MetaCausalWarning,
)

# Read from installed package metadata (pyproject.toml is the single source
# of truth) rather than hardcoding a second copy that can drift out of sync.
try:
    __version__ = _version("metacausal")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

warnings.warn(
    f"metacausal {__version__} is the final release and will not receive "
    "further fixes or features. Future development continues in xcausal "
    "(https://pypi.org/project/xcausal/).",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "CausalEnsemble",
    # Estimator result types
    "ComponentAteEstimate",
    "AteEstimate",
    "ComponentCateEstimate",
    "CateEstimate",
    # Aggregation
    "AggregationStrategy",
    "PointwiseStrategy",
    "AgreementStrategy",
    "Median",
    "Mean",
    "TrimmedMean",
    "CBA",
    "EnsembleWeights",
    "BootstrapResult",
    # Library adapters
    "CausalMLAdapter",
    "DoubleMLAdapter",
    "EconMLAdapter",
    "StochtreeAdapter",
    # Generic adapters
    "GenericATEAdapter",
    "GenericAdapter",        # backward-compatible alias
    "GenericCATEAdapter",
    # Outcome-type detection
    "infer_outcome_type",
    # Warning classes
    "MetaCausalWarning",
    "ComponentWarning",
    "ComponentFailureWarning",
    "ComponentExclusionWarning",
    "BootstrapWarning",
    # Backward-compatible aliases
    "CausalEstimate",
    "EnsembleEstimate",
]
