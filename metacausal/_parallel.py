"""Shared parallel-execution helpers.

Every site in the codebase that dispatches independent tasks to workers
should go through :func:`parallel_map` so the backend choice and thread
pinning stay consistent. See issue #10 for the "exclusive outer
parallelism" design.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable


_THREAD_PIN_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Sentinel env var set inside every parallel worker. Adapters check it to
#: force any joblib-parallel component (e.g. EconML's ``CausalForestDML``,
#: whose default ``n_jobs=-1`` would otherwise nest a second loky pool
#: inside this worker and oversubscribe cores) to run serially. The thread
#: pins above cover OpenMP/BLAS but *not* joblib's own ``n_jobs``, so this
#: has to be signalled separately. This is the "inner levels run
#: sequentially" half of the exclusive-outer-parallelism design.
INNER_WORKER_ENV = "METACAUSAL_INNER_WORKER"


def _pin_threads() -> None:
    """Pin inner threadpools to 1 thread in a worker process.

    Called as the ``initializer`` of every ``joblib.Parallel`` dispatch.
    Sets env vars for the OpenMP, OpenBLAS, MKL, numexpr, and Apple
    Accelerate runtimes. loky spawns fresh processes, so these take
    effect before any numeric library initializes its threadpool. Also
    sets :data:`INNER_WORKER_ENV` so adapters can pin joblib-based inner
    parallelism (which the thread vars do not reach).
    """
    for var in _THREAD_PIN_VARS:
        os.environ[var] = "1"
    os.environ[INNER_WORKER_ENV] = "1"


def force_serial(
    model: Any,
    attr_name: str = "n_jobs",
    sub_attrs: Iterable[str] | None = None,
) -> None:
    """Pin *attr_name* to 1 on *model* and its nested sub-models.

    Shared by adapters' ``_force_serial`` methods, called when fitting
    inside a MetaCausal worker (:data:`INNER_WORKER_ENV` set) to stop a
    wrapped estimator's own joblib-parallel knob (EconML's ``n_jobs``,
    CausalML's ``cv_n_jobs``, ...) from nesting a second worker pool inside
    the outer one and oversubscribing cores. Outside a worker, adapters
    leave the model's own setting untouched -- this function is only ever
    called from inside the ``INNER_WORKER_ENV`` branch.

    If *sub_attrs* is given, only those named attributes on *model* are
    inspected (a library-specific fixed set of known nested-model slots,
    e.g. EconML's ``model_y``/``model_t``/...). If omitted, every value in
    ``vars(model)`` is inspected instead, for libraries that store nested
    learners under varying names per class (e.g. CausalML's
    ``model_mu``/``model_tau``/``model_p`` vs. ``model_c``/``model_t``).
    Either way, list-valued attributes are walked item-by-item.
    """
    if hasattr(model, attr_name):
        setattr(model, attr_name, 1)

    candidates = (
        (getattr(model, a, None) for a in sub_attrs)
        if sub_attrs is not None
        else vars(model).values()
    )
    for sub in candidates:
        if sub is None:
            continue
        if isinstance(sub, list):
            for item in sub:
                if hasattr(item, attr_name):
                    setattr(item, attr_name, 1)
        elif hasattr(sub, attr_name):
            setattr(sub, attr_name, 1)


def parallel_map(
    n_jobs: int,
    func: Callable[..., Any],
    tasks: Iterable[tuple],
) -> list[Any]:
    """Dispatch ``func(*task)`` for each task, in parallel or sequentially.

    Uses the loky backend with :func:`_pin_threads` as worker initializer.
    Falls back to a sequential list comprehension when ``n_jobs == 1``
    to avoid joblib dispatch overhead and to keep stack traces readable
    during debugging.

    Parameters
    ----------
    n_jobs : int
        1 → sequential. -1 → all cores. Any other positive int → that many
        worker processes.
    func : callable
        Applied as ``func(*task)`` per task. Must be picklable.
    tasks : iterable of tuple
        Each tuple is unpacked as positional arguments to ``func``.

    Returns
    -------
    list
        Results in task order.
    """
    tasks = list(tasks)
    if n_jobs == 1 or len(tasks) <= 1:
        return [func(*t) for t in tasks]

    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [func(*t) for t in tasks]

    n_workers = n_jobs if n_jobs > 0 else os.cpu_count()
    return Parallel(
        n_jobs=n_workers,
        backend="loky",
        initializer=_pin_threads,
    )(delayed(func)(*t) for t in tasks)
