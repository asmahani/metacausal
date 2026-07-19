"""Internal text-formatting helpers for public result objects."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _auto_digits(x: float) -> int:
    x = abs(float(x))
    if x >= 100:
        return 0
    if x >= 1:
        return 1
    return 4


def format_scalar(
    x: float,
    *,
    digits: int | None = None,
    signed: bool = False,
) -> str:
    """Format a scalar for human-readable summaries."""
    if digits is None:
        digits = _auto_digits(x)
    spec = f"{'+' if signed else ''},.{digits}f"
    return format(float(x), spec)


def format_interval(
    lo: float | None,
    hi: float | None,
    *,
    digits: int | None = None,
    signed: bool = False,
) -> str | None:
    """Format a scalar interval, or return None if unavailable."""
    if lo is None or hi is None:
        return None
    return (
        f"[{format_scalar(lo, digits=digits, signed=signed)}, "
        f"{format_scalar(hi, digits=digits, signed=signed)}]"
    )


def summarize_array(x: np.ndarray, *, digits: int | None = None) -> str:
    """Return a compact descriptive summary of a 1-d array-like."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return "n=0"
    if digits is None:
        digits = _auto_digits(float(np.mean(np.abs(x))))
    return (
        f"n={x.size}, mean={format_scalar(float(np.mean(x)), digits=digits)}, "
        f"median={format_scalar(float(np.median(x)), digits=digits)}, "
        f"q10={format_scalar(float(np.quantile(x, 0.10)), digits=digits)}, "
        f"q90={format_scalar(float(np.quantile(x, 0.90)), digits=digits)}"
    )


def truncate_items(items: Iterable[str], *, max_items: int = 4) -> str:
    """Render an iterable of names, truncating after max_items."""
    items = list(items)
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + ", ..."


class SummaryStr(str):
    """A ``str`` whose REPL/notebook display matches ``print()``.

    Used as the return type of the package's various ``summary()``
    methods, which build multi-line, human-formatted text. A plain ``str``
    displays via ``repr()`` when typed bare at a prompt, which shows
    embedded newlines as literal ``\\n`` rather than rendering them --
    forcing users to remember to wrap every call in ``print(...)``. This
    subclass overrides only ``__repr__`` (to match ``__str__``), so bare
    evaluation looks the same as ``print()`` while every other ``str``
    behavior -- concatenation, formatting, equality, ``isinstance(x, str)``,
    etc. -- is unchanged.
    """

    def __repr__(self) -> str:
        return str(self)
