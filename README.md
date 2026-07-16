# MetaCausal

Cross-framework ensembling of causal machine-learning estimators for ATE and pointwise CATE, with full-pipeline bootstrap inference.

[![PyPI version](https://img.shields.io/pypi/v/metacausal)](https://pypi.org/project/metacausal/)
[![Python versions](https://img.shields.io/pypi/pyversions/metacausal)](https://pypi.org/project/metacausal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/asmahani/metacausal/blob/main/LICENSE)

## What it is

MetaCausal orchestrates multiple causal-ML estimators from different libraries — [EconML](https://github.com/py-why/EconML), [DoubleML](https://docs.doubleml.org), [CausalML](https://github.com/uber/causalml), [stochtree](https://stochtree.ai), or arbitrary user-supplied callables — behind a single protocol, and aggregates their treatment-effect estimates into a single ensemble estimate. Eight aggregation strategies are provided, grouped into three tiers:

- **Pointwise robust** — Median (default), Mean, Trimmed Mean.
- **Agreement-based** — Consensus Based Averaging, which selects a high-agreement subset of components from pairwise Kendall's τ.
- **Outcome-supervised** — Causal Stacking, R-Stacking, Q-Aggregation, and Select, which learn weights (or, for Select, choose a single best component) by optimising a causal loss on cross-fitted out-of-fold predictions.

A full-pipeline bootstrap supplies comparable confidence intervals for both ATE and pointwise CATE across heterogeneous components whose native inference machinery is otherwise incomparable.

## Why

No single causal-ML estimator dominates across data-generating processes, model selection for heterogeneous treatment effects is empirically unreliable, and individual methods can fail catastrophically under specific violations of their own assumptions (overlap breakdown, nuisance misspecification, tree extrapolation). MetaCausal's default pointwise median aggregation gives a 50% breakdown point with no tuning — up to half the component estimators can produce arbitrarily bad estimates without corrupting the ensemble. When outcome data allow learning weights, MetaCausal also ships the three outcome-supervised stackers from the recent CATE-ensemble literature, plus Select as a feasible-selection baseline for comparison.

## Installation

```bash
pip install metacausal
```

This installs the core package and its required dependencies (`numpy`, `pandas`, `scipy`). Estimator libraries are optional extras:

```bash
# Individual libraries
pip install "metacausal[econml]"
pip install "metacausal[doubleml]"
pip install "metacausal[causalml]"
pip install "metacausal[stochtree]"

# Visualisation helpers (matplotlib)
pip install "metacausal[plots]"

# Everything (frameworks + plots)
pip install "metacausal[all]"
```

Python 3.11 or later is required.

## Quick start

```python
from metacausal import CausalEnsemble
from metacausal.datasets import load_lalonde

X, T, Y = load_lalonde()

# Default ensemble: nine estimators spanning EconML, DoubleML,
# CausalML, and stochtree, aggregated by pointwise median.
ens = CausalEnsemble()
ens.fit(X, T, Y, random_state=42)

# Point estimate. summary() prints the ensemble ATE, the per-component
# table, and the spread; the raw values stay on the object (ate.ate, ...).
ate = ens.ate()
print(ate.summary())

# Full-pipeline bootstrap confidence interval
boot = ens.bootstrap(n_boot=200, random_state=42, n_jobs=-1)
print(boot.summary())
```

The three-step `fit → ate / cate → bootstrap` pattern is the recommended one, because it lets you inspect intermediate state and swap aggregation strategies on an already-fitted ensemble. The convenience wrapper `ens.estimate(X, T, Y, n_boot=200, ...)` does `fit + bootstrap` (or `fit + ate`) in a single call.

## Aggregation strategies at a glance

| Tier | Strategy | String alias / class | Data used |
|---|---|---|---|
| Pointwise | **Median** *(default)* | `"median"` / `Median` | Component predictions only |
| Pointwise | Mean | `"mean"` / `Mean` | Component predictions only |
| Pointwise | Trimmed Mean | `"trimmed_mean"` / `TrimmedMean` | Component predictions only |
| Agreement | Consensus Based Averaging | `"cba"` / `CBA` | Component CATE predictions on training data |
| Supervised | Causal Stacking | `CausalStacking` | Cross-fitted OOF predictions + nuisance |
| Supervised | R-Stacking | `RStacking` | Cross-fitted OOF predictions + nuisance |
| Supervised | Q-Aggregation | `QAggregation` | Cross-fitted OOF predictions + nuisance |
| Supervised | Select (best-by-risk) | `Select` | Cross-fitted OOF predictions + nuisance |

```python
# By string alias (default configuration)
ens = CausalEnsemble(aggregation="trimmed_mean")

# By object (lets you configure hyperparameters)
from metacausal.aggregation import QAggregation
ens = CausalEnsemble(aggregation=QAggregation(nu=0.5, greedy=True))
```

See the accompanying paper (forthcoming) for the mathematical details of each strategy.

## Outcome types

MetaCausal supports two outcome types:

- **Continuous** *(default)* — any numeric Y not detected as binary. Quietly absorbs counts, bounded continuous, and ordinal-as-numeric. The base learner choice is the user's responsibility (an `HistGradientBoostingRegressor` by default; user-supplied components can pass a Poisson booster if appropriate).
- **Binary** — numeric Y with values ⊆ {0, 1} or boolean dtype. The estimand is the **risk difference** ATE/CATE (mean difference of probabilities).

Detection happens at fit time: `CausalEnsemble().fit(X, T, Y)` inspects Y, picks the right pool from `default_methods`, and routes nuisance estimation through `predict_proba` for binary outcomes. To force an interpretation, pass `outcome_type="continuous"` or `outcome_type="binary"` at construction. Multi-class / nominal and survival outcomes are out of scope; encoding-as-multiple-binary or a dedicated survival library is the recommended path.

The default binary pool (8 components) drops `DoubleMLPLR`, EconML S/T/X-Learners, the `BaseRRegressor`, and stochtree BCF — all of which either lack a binary-capable code path in their upstream library or would silently fit a linear-probability model — and substitutes the CausalML classifier siblings (`BaseSClassifier`, `BaseTClassifier`, `BaseXClassifier`, `BaseRClassifier`). `DoubleMLIRM`, `CausalForestDML`, `DRLearner`, and `TMLELearner` remain.

## Usage recipes

Snippets below that use `X`, `T`, `Y` without redefining them assume the
`load_lalonde()` call from [Quick start](#quick-start); snippets that need
different data (e.g. a binary outcome) load it explicitly.

### Mixed-framework method list

Estimators from EconML and CausalML are auto-detected by module prefix; DoubleML, stochtree, and arbitrary callables go through explicit adapters.

```python
from metacausal import CausalEnsemble, GenericATEAdapter
from metacausal.adapters import DoubleMLAdapter, CausalMLAdapter
from econml.dml import CausalForestDML
from econml.metalearners import TLearner, XLearner
from doubleml import DoubleMLIRM
from causalml.inference.meta import BaseDRRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor as HGBR,
    HistGradientBoostingClassifier as HGBC,
)

def naive_diff(X, T, Y):
    return float(Y[T == 1].mean() - Y[T == 0].mean())

ens = CausalEnsemble(
    methods=[
        CausalForestDML(discrete_treatment=True),              # auto-wrapped (EconML)
        TLearner(models=HGBR()),                               # auto-wrapped (EconML)
        XLearner(models=HGBR(), propensity_model=HGBC()),      # auto-wrapped (EconML)
        DoubleMLAdapter(DoubleMLIRM, ml_g=HGBR(), ml_m=HGBC()),
        CausalMLAdapter(BaseDRRegressor(learner=HGBR())),
        GenericATEAdapter(naive_diff, name="naive_diff"),
    ],
    aggregation="median",
)
ens.fit(X, T, Y, random_state=42)
print(ens.ate().ate)
```

To configure **analytical** upstream inference, wrap the estimator explicitly instead of relying on auto-detection:

```python
from causalml.inference.meta import BaseTRegressor
from doubleml import DoubleMLIRM
from econml.dml import CausalForestDML
from metacausal.adapters import DoubleMLAdapter, EconMLAdapter, CausalMLAdapter, StochtreeAdapter
from sklearn.ensemble import (
    HistGradientBoostingRegressor as HGBR,
    HistGradientBoostingClassifier as HGBC,
)

dml = DoubleMLAdapter(DoubleMLIRM, ml_g=HGBR(), ml_m=HGBC(), alpha=0.10)
econ = EconMLAdapter(CausalForestDML(model_y=HGBR(), model_t=HGBC(), discrete_treatment=True), alpha=0.10, inference="statsmodels")
cml = CausalMLAdapter(BaseTRegressor(learner=HGBR(), ate_alpha=0.10))  # upstream CausalML control
st = StochtreeAdapter(alpha=0.10)
```

For CausalML, analytical **ATE** CI settings stay on the wrapped upstream estimator (`ate_alpha` on meta-learners / TMLE, `alpha` on `CausalTreeRegressor`); the adapter does not override them.

### Binary outcome on real data

`load_lalonde(binarize_y=...)` returns the 1978-earnings outcome as a binary indicator — `"median"` for the (~50/50) above-median split, `"positive"` for the (~69/31) "any 1978 earnings" indicator. Useful as a real-data fixture without leaving the package.

```python
from metacausal import CausalEnsemble
from metacausal.datasets import load_lalonde

X, T, Y = load_lalonde(binarize_y="median")

# outcome_type="auto" detects binary Y, materialises the binary
# default pool (8 components targeting the risk difference), and
# fits. ATE is on the risk-difference scale, in [-1, 1].
ens = CausalEnsemble()
ens.fit(X, T, Y, random_state=42)

print(ens.ate().ate)
```

### CATE estimation with a supervised strategy

```python
from metacausal import CausalEnsemble
from metacausal.aggregation import CausalStacking

ens = CausalEnsemble(aggregation=CausalStacking())
ens.fit(X, T, Y, random_state=42)

# Pointwise CATE CIs at X_eval (any covariate matrix; omit to
# evaluate at the training X instead)
X_eval = X
boot = ens.bootstrap(X_eval, n_boot=200, random_state=42, n_jobs=-1)

print(boot.cate)           # ensemble CATE at X_eval, shape (n_eval,)
print(boot.cate_ci_lower)  # pointwise 95% lower bound
print(boot.cate_ci_upper)  # pointwise 95% upper bound

# Inspect the learned ensemble weights
print(boot.ensemble_weights)
```

### Compare aggregation strategies without refitting

An `aggregation=...` argument to `ate()` or `cate()` re-aggregates from cached predictions without refitting components — useful for quick comparisons.

```python
ens = CausalEnsemble(aggregation="median")
ens.fit(X, T, Y, random_state=42)

for agg in ["median", "mean", "trimmed_mean", "cba"]:
    ate = ens.ate(aggregation=agg)
    print(f"{agg:<15} ATE = {ate.ate:.1f}")
```

### Visualisation helpers

The optional `metacausal.plots` submodule (installed via the `[plots]` extra) provides four matplotlib helpers that consume the result types above:

- `forest(boot)` — component and ensemble ATEs with bootstrap CIs.
- `weights(ens)` — aggregation weight bars (agreement-based and supervised strategies).
- `cate_profile(source, x, xlabel=...)` — ensemble CATE along one covariate, with optional bootstrap band and per-component overlay.
- `disagreement(ens, X)` — pairwise component-CATE rank-correlation heatmap.

```python
from metacausal.plots import forest, cate_profile

forest(boot)
grid = X[:, 6]  # `re74`; see load_lalonde()'s docstring for column order
cate_profile(boot, x=grid, xlabel="re74 (1974 earnings, USD)")
```

## Extending MetaCausal

MetaCausal exposes five injection points that let researchers extend the package without forking it: custom component adapters, custom aggregation strategies, replacement nuisance pipelines (`fit_nuisance_fn`), replacement pseudo-outcome functions (`pseudo_outcome_fn`), and custom cross-fitting splitters. The accompanying paper (forthcoming) covers each injection point in detail.

The lowest-effort path for adding a new estimator is `GenericCATEAdapter`, which wraps a fit function, a CATE prediction function, and (optionally) an ATE prediction function into a component without implementing the full protocol:

```python
from metacausal import CausalEnsemble, GenericCATEAdapter

def fit_fn(X, T, Y, **kwargs):
    # Train your model and return any state you need.
    ...
    return state

def cate_fn(state, X):
    # Return per-observation CATE estimates, shape (n,).
    return state.predict_cate(X)

def ate_fn(state, X):  # optional; defaults to mean of cate_fn(state, X)
    return float(cate_fn(state, X).mean())

my_method = GenericCATEAdapter(
    fit_fn, cate_fn, fn_ate=ate_fn, name="my_method",
)

ens = CausalEnsemble(methods=[my_method, ...])
```

**Parallelism cooperation contract.** MetaCausal automatically suppresses the internal `n_jobs`/`cv_n_jobs` of the estimators it wraps for EconML, CausalML, DoubleML, and stochtree whenever they run inside one of MetaCausal's own workers (`fit`/`bootstrap`, or supervised cross-fitting), so those defaults can't nest a second worker pool and oversubscribe cores. That automatic suppression can't reach into an opaque `fn_fit`/`fn_cate`/`fn_ate` callable — if your model parallelizes internally, either construct it with its own parallelism knob fixed at `1` up front, or check the same signal MetaCausal's built-in adapters use:

```python
import os
from metacausal.adapters import INNER_WORKER_ENV

def fit_fn(X, T, Y, **kwargs):
    model = MyParallelModel(n_jobs=-1)
    if os.environ.get(INNER_WORKER_ENV):
        # Already inside a MetaCausal worker; nesting our own pool here
        # would oversubscribe cores, so fall back to serial.
        model.n_jobs = 1
    model.fit(X, T, Y)
    return model
```

## Reproducibility and parallelism

A single `random_state` seed deterministically propagates to every stochastic sub-step — component models, their sub-estimators, cross-fitting folds, nuisance fits, and bootstrap replicates — so reruns are bit-identical.

A single `n_jobs` knob on `fit`, `bootstrap`, and `estimate` routes parallelism to the outermost applicable level (bootstrap replicates when `n_boot > 0`; otherwise supervised cross-fitting or component fits) and pins BLAS/OpenMP threads inside each worker to prevent oversubscription. The accompanying paper (forthcoming) explains the rationale.

This exclusive-outer-parallelism design also covers the estimators MetaCausal wraps, not just its own workers: for EconML, CausalML, DoubleML, and stochtree components, MetaCausal suppresses the wrapped estimator's own internal parallelism knob (EconML/DoubleML's `n_jobs`, CausalML's `cv_n_jobs`) whenever it runs inside one of MetaCausal's workers, as an ongoing guarantee rather than a one-off fix. Custom components (`GenericCATEAdapter`, arbitrary callables) are not covered automatically — see "Extending MetaCausal" above for the cooperation contract if your own model parallelizes internally.

The outer process (your main script) keeps the platform-default BLAS thread count, which is fine on macOS and Windows. On Linux, where joblib's loky backend can occasionally deadlock at fork time when the parent's BLAS pool is already running threads, defensive users may want to set the standard thread env vars (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`) before invoking Python. The bundled replication runner and the test suite's `tests/conftest.py` set these automatically, so reviewers and contributors do not need the shell prefix.

```python
# Parallelise supervised cross-fitting, deterministic:
ens = CausalEnsemble(aggregation=CausalStacking())
ens.fit(X, T, Y, random_state=42, n_jobs=-1)

# Or: full fit + bootstrap pipeline with bootstrap-level parallelism:
boot = ens.estimate(X, T, Y, n_boot=500, random_state=42, n_jobs=-1)
```

## Citation

A BibTeX entry will be added here when the arXiv preprint of the accompanying manuscript is posted. For interim references to the software itself, see the PyPI listing.

## Further reading

- **Paper:** a preprint covering the methodology, architecture, and extensibility hooks is in preparation. An arXiv link will be added here once it is posted.
- **Replication material:** will be included as ancillary files with the forthcoming arXiv submission.

## Release notes

See [CHANGELOG.md](CHANGELOG.md) for the full version history. Latest: **0.7.0 (Unreleased)** — a new `Select` aggregation strategy, a core-oversubscription fix for CausalML's R-Learner, plotting functions now also reachable as methods, a Sphinx API reference, and several non-executable docstring/README example fixes.

## License

MetaCausal is distributed under the MIT License. See [LICENSE](https://github.com/asmahani/metacausal/blob/main/LICENSE).
