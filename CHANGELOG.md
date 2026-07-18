# Changelog

All notable changes to MetaCausal are documented here. Dates are
release dates; "Unreleased" entries describe changes on `main` not yet
tagged.

## Unreleased

- **Widened the `causalml`, `stochtree`, and `doubleml` caps** to admit each library's current latest patch/minor (`causalml<0.17.1`, `stochtree<0.4.6`, `doubleml<0.11.4`), validating CI (including `pytest -m integration`) against `causalml==0.17.0`, `stochtree==0.4.5`, and `doubleml==0.11.3`. Floors are unchanged; per the versioning policy in `pyproject.toml`, only the cap moves until the floor itself is deliberately re-validated and bumped.

## 0.7.0 — 2026-07-17

- **New aggregation strategy: `Select`.** Chooses the single component model minimizing a pseudo-outcome risk (`loss="dr"` for DR/AIPW plug-in risk, `loss="r"` for R-risk on the Robinson residuals) instead of combining components. Implemented as a `SupervisedStrategy` so it shares cross-fitted nuisance estimation, `EnsembleWeights` introspection, and bootstrap with the true aggregators — it returns a one-hot weight vector rather than a blend. `Select(loss="dr")` is the closed-form vertex of `QAggregation`'s `nu=1` limit, computed directly by argmin rather than via the general solver. Serves as the feasible-selection comparator for ensemble-vs-selection studies.
- **Fixed core/process oversubscription from CausalML's R-Learner default pool component.** CausalML's `BaseRLearner` family (`BaseRRegressor`/`BaseRClassifier`, in the default continuous/binary pools) defaults `cv_n_jobs=-1` for its internal `cross_val_predict` call — unlike EconML's `n_jobs`, which most wrapped estimators default to serial, this knob is *on* by default, so left alone it spawned a full-core joblib pool on every single fit, nesting inside MetaCausal's own outer parallelism (`fit()`/`bootstrap()`, or an external harness's own outer parallel loop) and oversubscribing cores. The shipped default pools now pin `cv_n_jobs=1` at construction, and `CausalMLAdapter` gates the wrapped model's `cv_n_jobs` on `METACAUSAL_INNER_WORKER` (mirroring `EconMLAdapter`'s existing `n_jobs` suppression) so user-supplied R-Learner components get the same protection. The two adapters' pinning logic is now shared via `metacausal._parallel.force_serial`.
- **Plotting functions are now also methods.** `forest`, `weights`, and `cate_profile` (from `metacausal.plots`) each stay the primary, documented implementation, but are now also reachable as thin delegating methods on the class they act on: `BootstrapResult.forest()`, `BootstrapResult.cate_profile(x, ...)`, `CateEstimate.cate_profile(x, ...)`, `EnsembleWeights.plot()`, `CausalEnsemble.weights()`, and `CausalEnsemble.disagreement(X)`. Both surfaces stay in sync — the methods just call the functions — and matplotlib remains an optional dependency imported only when one of these six methods (or functions) is actually called.
- **Added a Sphinx API reference, published at [asmahani.github.io/metacausal](https://asmahani.github.io/metacausal/).** Full public-API coverage via `autodoc`/`napoleon`/`autosummary` — `CausalEnsemble`, every result/estimate class, `metacausal.aggregation`, `metacausal.adapters`, `metacausal.plots`, `metacausal.datasets` — built with the `furo` theme, and checked in CI with `sphinx-build -W` (warnings-as-errors) plus the `doctest` builder. Deployed to GitHub Pages automatically on every push to `main`. Build locally with `pip install -e ".[all,docs]"` then `python -m sphinx -b html docs docs/_build/html`.
- **Fixed three non-executable docstring examples.** `CausalEnsemble`, `DoubleMLAdapter`, and `StochtreeAdapter` each had an `Example::` code block that referenced undefined placeholder data (`X_train`/`T_train`/`Y_train` or `X`/`T`/`Y`), and `CausalEnsemble`'s additionally passed `DoubleMLIRM` the wrong kwarg (`ml_l=`, which is PLR-specific; IRM takes `ml_g=`). All three are now self-contained, fast, and written as real `>>>` doctests that the new `docs` CI job actually executes.
- **Fixed two non-executable README snippets.** The Visualisation helpers recipe referenced an undefined `grid`, and the CATE-with-a-supervised-strategy recipe referenced an undefined `X_eval`; both are now defined inline. Every Python code block in this README (except the intentionally-skeletal `GenericCATEAdapter` template) has been run end to end against this release. Added a note at the top of "Usage recipes" clarifying that snippets reusing `X`, `T`, `Y` without redefining them assume the Quick start data.
- **Published the custom-component parallelism cooperation contract.** `metacausal.adapters.INNER_WORKER_ENV` (the sentinel env var custom components can check to cooperate with MetaCausal's own-worker parallelism guard) is now a public, documented export, not just an underscore-prefixed internal in `metacausal._parallel`. Documented the cooperation contract in `GenericCATEAdapter`'s docstring and README's "Extending MetaCausal" section (with a worked example), and added a sentence to "Reproducibility and parallelism" stating known-adapter `n_jobs`/`cv_n_jobs` suppression (EconML, CausalML, DoubleML, stochtree) as an ongoing guarantee rather than a changelog-only artifact.

## 0.6.1 — 2026-07-01

- **Documentation:** the README quick start and recipes now use the result objects' `summary()` methods and the `EnsembleWeights` repr (from 0.6.0) in place of hand-formatted loops; corrected the stated default continuous-pool size (nine, not ten) and aligned the bootstrap wording with the rest of the docs ("full-pipeline", not "honest").

## 0.6.0 — 2026-06-30

- **Human-readable result summaries:** `CausalEnsemble` and the main result objects now expose compact `__repr__` output, and `AteEstimate`, `CateEstimate`, and `BootstrapResult` now provide explicit `summary()` methods for terminal-friendly multi-line reports without dumping full NumPy arrays. `summary()` accepts `digits=` and `signed=` for column formatting; `AteEstimate.summary()` additionally accepts `show_ci=False` to omit the per-component native confidence intervals (cleaner point-estimate table when CIs are unavailable for some methods or reported separately).
- **EconML/sklearn deprecation suppression:** `EconMLAdapter` now suppresses sklearn's `FutureWarning` about `force_all_finite` being renamed to `ensure_all_finite` around wrapped EconML prediction and analytical-interval calls (`ate`, `cate`, and their fallback/effect paths). The filter is adapter-local, narrowed by category + message + module, and scoped to the single call so unrelated `FutureWarning`s still surface.

## 0.5.0 — 2026-06-15

- **`bootstrap()` flags a point estimate outside its CI under both resampling schemes.** The `BootstrapWarning` for an ATE confidence interval that does not contain the point estimate previously fired only under the `subsample` scheme; it now fires under `nonparametric` too, with a scheme-specific explanation. Containment is governed by the same percentile condition for both schemes — the √(m/n) scaling changes only the interval width — and with-replacement resampling (~63% distinct units) or per-replicate weight re-optimization can shift the replicate distribution off the point estimate. Pre-1.0 behavior change: nonparametric bootstraps that previously ran silently may now emit a warning.
- **Single-valued outcomes are rejected with an actionable error.** `infer_outcome_type` previously classified a constant `Y` (all zeros, all ones, or any single repeated value) as `binary`, which surfaced later as an opaque scikit-learn "needs samples of at least 2 classes" failure. It now raises a clear `ValueError` up front. Pre-1.0 behavior change.
- **Docstring corrections.** The supervised aggregation strategies (`CausalStacking`, `RStacking`, `QAggregation`) documented a stale `LGBMRegressor`/`LGBMClassifier` default and omitted the regressor-vs-classifier requirement for binary outcomes; they now describe the `HistGradientBoosting*` defaults and the outcome-type-dependent model contract. `GenericCATEAdapter` no longer references a non-existent extensibility guide, and its `fn_ate` example now performs a genuine doubly robust (AIPW) computation instead of silently reproducing the default `mean(cate(X))`.
- **Correction to the 0.4.0 parallel-safety note.** The intermittent segmentation faults in EconML's Cython tree builder (`CausalForestDML`) are a latent out-of-bounds bug in EconML's generalized random forest — not merely loky oversubscription. They reproduce single-threaded, surfacing under the repeated component refits a bootstrap performs ([EconML #470](https://github.com/py-why/EconML/issues/470), unresolved as of econml 0.16.0). The 0.4.0 `n_jobs=1` pin is kept as defensive hardening (it removes genuine oversubscription and lowers crash probability) but does not fully eliminate the crash: `bootstrap()` with `CausalForestDML` in the pool can still segfault intermittently under heavy parallelism.

## 0.4.0 — 2026-06-07

- **Parallel-safety fix:** the default `CausalForestDML` now builds its forest with `n_jobs=1`, and `EconMLAdapter` pins any wrapped joblib-parallel estimator to a single job when it runs inside one of MetaCausal's own parallel workers (`fit`, `bootstrap`, or supervised cross-fitting). EconML defaults `CausalForestDML` to `n_jobs=-1`, whose inner loky pool would otherwise nest inside the outer worker and intermittently segfault EconML's Cython tree builder under oversubscription. Outputs are unchanged — the forest is seeded deterministically — only the nested parallelism is removed.
- **Default pool update:** S-Learner and its classifier sibling (`BaseSClassifier`) are now excluded from the default pools for continuous and binary outcomes, respectively; the default pools are now nine and seven components.

## 0.3.1 — 2026-06-05

- **Dependency declaration:** `joblib>=1.2` is now a direct core dependency instead of arriving only transitively through `scikit-learn`.
- **Top-level adapter imports:** `DoubleMLAdapter`, `EconMLAdapter`, and `StochtreeAdapter` are now exported from `metacausal`, so users no longer need submodule import paths for those adapters.
- **Custom CATE shorthand:** `CausalEnsemble` now auto-wraps 2- and 3-callable lists/tuples as `GenericCATEAdapter`, allowing `(fn_fit, fn_cate)` and `(fn_fit, fn_cate, fn_ate)` directly in `methods=[...]`.

## 0.3.0 — 2026-05-30

- **Analytical CI controls:** `DoubleMLAdapter(alpha=...)` now forwards to `DoubleML.confint(level=1 - alpha)`, and `EconMLAdapter(alpha=..., inference=...)` now forwards both the interval level and the fit-time inference backend.
- **CausalML clarification:** analytical ATE CI level was already configurable upstream via the wrapped estimator (`ate_alpha` / estimator-specific `alpha`); the docs now describe that accurately instead of implying a missing adapter feature.
- **API cleanup:** removed the unused `alpha` argument from `CausalEnsemble.cate()`. Analytical component-level CI configuration now lives on the relevant adapter or wrapped upstream estimator, while ensemble CI level remains on `bootstrap(alpha=...)`.

## 0.2.2 — 2026-05-04

- **Dependency hygiene:** the four causal-ML extras (`econml`, `doubleml`, `stochtree`, `causalml`) are now pinned to a single patch each — floors at the exact version validated by CI on the most recent main-branch run, caps at the next patch. Motivated by [stochtree #376](https://github.com/StochasticTree/stochtree/issues/376), where a patch release (0.4.0 → 0.4.2) silently changed the semantics of `BCFModel.predict(terms="tau")` and broke 0.2.0. Patch-level caps mean every upstream release lands outside the cap, triggers a Dependabot PR, and runs `pytest -m integration` before we widen — closing the silent-install hole that produced the 0.2.1 hotfix.

## 0.2.1 — 2026-05-04

- **Bug fix:** `StochtreeAdapter` now calls `BCFModel.predict(..., terms="cate")` instead of `terms="tau"`. With `stochtree 0.4.2` (which added a parametric treatment-intercept term in the BCF sampler), `terms="tau"` returned the forest-only piece and excluded the parametric component, producing wildly seed-sensitive ATEs that disagreed sharply with the rest of the default ensemble. `terms="cate"` returns the full conditional treatment effect — including parametric and random-slope components, when present — for any BCF configuration. Fixes upstream issue [stochtree #376](https://github.com/StochasticTree/stochtree/issues/376) on the metacausal side.

## 0.2.0 — 2026-05-04

**New**

- Outcome-type handling: `CausalEnsemble` auto-detects continuous vs binary `Y` at `fit()`, materialises the right default pool, and routes nuisance through `predict_proba` for binary. Override via `outcome_type="continuous"|"binary"`. Public `metacausal.infer_outcome_type(Y)` utility; `binarize_y={"median","positive"}` on `load_lalonde()`.
- Subsample bootstrap (`bootstrap(method="subsample")`): m-out-of-n without replacement, T-stratified, with Politis–Romano scaled-percentile CIs. Eliminates duplicate-unit leakage across cross-fit folds.
- Structured warning hierarchy: `ComponentFailureWarning`, `ComponentExclusionWarning`, `BootstrapWarning` under a common `MetaCausalWarning` umbrella.
- `CausalMLAdapter` accepts a `propensity_model=` kwarg, forwarding a fitted propensity to non-TMLE meta-learners.

**Breaking changes for custom-strategy / custom-adapter authors**

- `AggregationStrategy` and family are `abc.ABC` with a unified `aggregate` entry point. Subclasses now implement `aggregate` rather than per-mode methods.
- Every adapter must declare `supported_outcome_types` and implement `validate_outcome_type(detected)`; the injectable `fit_nuisance_fn` gains an `outcome_type` parameter.

**Other**

- Bounded version constraints on `econml`, `doubleml`, `causalml`, `stochtree` (capped at next minor; floors anchored to tested versions).
- `requires-python` raised to `>=3.11` (causalml 0.16 floor).
- Tier-2 integration tests via `pytest -m integration`.
- Bug fixes: `load_lalonde` no longer leaks a file handle; `EconMLAdapter` suppresses the upstream `DataConversionWarning` from `DRLearner(discrete_outcome=True)`.
- PyPI metadata polish (classifiers, license badge).

Tested against: `doubleml 0.11.2`, `econml 0.16.0`, `causalml 0.16.0`, `stochtree 0.4.0`.

## 0.1.0 — 2026-04-25

Initial public release.
