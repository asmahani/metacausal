"""Adapters for wrapping causal ML libraries."""

from metacausal._parallel import INNER_WORKER_ENV as _INNER_WORKER_ENV
from metacausal.adapters.base import CausalEstimator
from metacausal.adapters.causalml import CausalMLAdapter, _is_causalml
from metacausal.adapters.doubleml import DoubleMLAdapter
from metacausal.adapters.econml import EconMLAdapter
from metacausal.adapters.generic import GenericAdapter, GenericATEAdapter, GenericCATEAdapter
from metacausal.adapters.stochtree import StochtreeAdapter

#: Sentinel env var set inside every MetaCausal parallel worker. Custom
#: components (:class:`~metacausal.adapters.GenericCATEAdapter`, arbitrary
#: callables) can check this to cooperate with MetaCausal's own-worker
#: parallelism guard -- pin your model to serial when it is set, the same
#: way EconMLAdapter/CausalMLAdapter/... do internally. See "Extending
#: MetaCausal" in the project README.
INNER_WORKER_ENV = _INNER_WORKER_ENV

__all__ = [
    "CausalEstimator",
    "CausalMLAdapter",
    "DoubleMLAdapter",
    "EconMLAdapter",
    "GenericATEAdapter",
    "GenericAdapter",        # backward-compatible alias for GenericATEAdapter
    "GenericCATEAdapter",
    "StochtreeAdapter",
    "INNER_WORKER_ENV",
]
