"""Shared constants and helpers for metacausal.plots.

Colour conventions are kept in one place so that every figure in the
submodule uses the same ensemble/component palette. Callers who want a
different palette can pass their own ``ax`` with a customised style
context.
"""

from __future__ import annotations

# Colour conventions used across every figure.
ENSEMBLE_COLOR = "C0"
COMPONENT_COLOR = "gray"
ZERO_REFERENCE_COLOR = "black"
