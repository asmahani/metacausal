"""Pytest configuration for metacausal.

The only thing this file does today is pin BLAS / OpenMP threadpools
to one thread *before* any test module imports numpy / sklearn / etc.
Pytest discovers conftest.py at session start and runs it before
collecting tests, so setting these env vars at the top — before any
other imports — guarantees they take effect before the threadpools
are initialised.

Without this, the test suite takes ~65 s on a typical workstation
because the per-fit thread spawn overhead dominates the small-sample
fits the suite is dominated by; with pinning, ~18 s. Mirrors the
shell-prefix recommended in CLAUDE.md, but applied automatically
so contributors don't need to type the prefix every time.

``setdefault`` so a contributor who deliberately wants more threads
(e.g., for benchmarking) can override via the shell.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
