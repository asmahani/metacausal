"""Base protocol for causal estimators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from metacausal.estimators import ComponentAteEstimate, ComponentCateEstimate


@runtime_checkable
class CausalEstimator(Protocol):
    """Protocol that all causal estimator adapters must satisfy.

    Any object with a ``name`` property, ``supports_cate`` flag, the
    ``supported_outcome_types`` declaration, and ``fit``/``ate``/``cate``
    /``validate_outcome_type`` methods matching this signature can be
    used as a component in :class:`CausalEnsemble`.

    Outcome-type capability is declared via two members:

    - ``supported_outcome_types``: a tuple of strings drawn from
      ``("continuous", "binary")`` indicating which outcome types this
      adapter can handle. The ensemble filters the candidate pool against
      the detected outcome type using this declaration: components whose
      tuple lacks the detected type are dropped before fitting.
    - ``validate_outcome_type(detected)``: raises ``ValueError`` if the
      adapter's *configured* nuisance learners are inconsistent with the
      detected outcome type — for example, a regressor wired into a slot
      that needs a classifier when ``Y`` is binary. For adapters with no
      user-configurable nuisance, a no-op is appropriate.
    """

    @property
    def name(self) -> str: ...

    @property
    def supports_cate(self) -> bool: ...

    supported_outcome_types: tuple[str, ...]

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        **kwargs,
    ) -> None: ...

    def ate(self, X: np.ndarray | None = None) -> ComponentAteEstimate: ...

    def cate(self, X: np.ndarray) -> ComponentCateEstimate: ...

    def validate_outcome_type(self, detected: str) -> None: ...
