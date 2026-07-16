"""Outcome-type detection for causal inference.

Determines whether an outcome vector ``Y`` is binary or continuous. Rejects
non-numeric, NaN, and other unsupported inputs with actionable messages.

The taxonomy is intentionally minimal: only ``"continuous"`` and
``"binary"`` are supported. Multi-class / nominal and survival outcomes
are out of scope — users should encode multi-class as one-vs-rest or use
a dedicated survival library.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def infer_outcome_type(Y) -> Literal["continuous", "binary"]:
    """Detect the outcome type from the values of ``Y``.

    Detection is on the *value set*, not cardinality: continuous ``Y``
    with only two distinct values is correctly classified as continuous
    unless those values are exactly ``{0, 1}``. Boolean ``Y`` is silently
    treated as binary. A single-valued ``Y`` (only one distinct value) is
    rejected, since treatment-effect estimation needs at least two levels.

    Parameters
    ----------
    Y : array-like
        Outcome vector. NumPy array, list, pandas Series, etc.

    Returns
    -------
    "binary" or "continuous"

    Raises
    ------
    ValueError
        If ``Y`` has non-numeric dtype, contains NaN, is empty, has only
        one distinct value, or otherwise cannot be unambiguously classified.
    """
    Y_arr = np.asarray(Y)

    if Y_arr.size == 0:
        raise ValueError("Outcome Y is empty.")

    if Y_arr.dtype.kind in ("O", "U", "S"):
        raise ValueError(
            f"Outcome Y has non-numeric dtype {Y_arr.dtype!r}. "
            "Categorical, string, or survival outcomes are not supported. "
            "If your outcome is binary, encode it as integer 0/1; if "
            "multi-class, encode as one-vs-rest and fit one ensemble per "
            "contrast."
        )

    if Y_arr.dtype == bool:
        return "binary"

    if not np.issubdtype(Y_arr.dtype, np.number):
        raise ValueError(
            f"Outcome Y has unsupported dtype {Y_arr.dtype!r}. Y must be "
            "numeric (integer, float, or boolean)."
        )

    if np.issubdtype(Y_arr.dtype, np.floating) and np.isnan(Y_arr).any():
        raise ValueError(
            "Outcome Y contains NaN values. Drop or impute missing "
            "outcomes before fitting."
        )

    unique = np.unique(Y_arr)
    if len(unique) < 2:
        raise ValueError(
            f"Outcome Y has only one distinct value ({unique.tolist()}). "
            "Treatment-effect estimation requires at least two outcome "
            "levels; check for a constant or degenerate outcome."
        )
    if set(unique.tolist()) == {0, 1}:
        return "binary"

    return "continuous"
