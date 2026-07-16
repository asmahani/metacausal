# MetaCausal — Project Notes

## Testing

When running `pytest`, always pin BLAS/OpenMP threadpools to 1 thread.
Without pinning, the test suite takes ~65 s; with pinning, ~18 s — a
3.6× difference driven entirely by thread-spawn overhead on small-sample
fits.

The package itself enforces this *inside* its own parallel workers (via
`metacausal/_parallel.py:_pin_threads` registered as the `joblib`
initializer), but pytest is an outer harness that bypasses that path
when it dispatches to its own xdist workers or imports estimators
directly. The pinning therefore has to be applied at the shell level,
before Python starts.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    .venv/bin/pytest -q
```

The same prefix applies to `pytest -n auto` if running parallel; in
practice serial+pinned beats parallel+pinned on this codebase because
the suite is dominated by short fits where joblib/xdist dispatch
overhead exceeds the parallel speedup.

Also: use `.venv/bin/pytest`, not the system `python3 -m pytest` —
package extras like `econml`, `causalml`, `doubleml`, `stochtree` are
installed only inside the project virtualenv.

## Git

When creating git commits for this repository, do not include a
Copilot/AI co-author trailer unless explicitly requested.
