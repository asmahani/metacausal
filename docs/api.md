# API reference

One page per public class and function, generated from docstrings via
`sphinx.ext.autosummary`. Each `##` section below is a real import
path (`metacausal`, `metacausal.aggregation`, ...); each item is
listed once, under the module it's *defined* in.

Many aggregation-strategy and adapter classes (e.g. `Median`,
`CausalMLAdapter`) are re-exported at the top level for convenience --
`from metacausal import Median` works even though it's documented
under `metacausal.aggregation` below. See the [Mixed-framework method
list](index.md#mixed-framework-method-list) or `metacausal.__all__`
for the exact top-level surface.

## `metacausal`

Top-level namespace: the ensemble class, its estimate/result types,
and outcome-type detection.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.CausalEnsemble
   metacausal.AteEstimate
   metacausal.CateEstimate
   metacausal.ComponentAteEstimate
   metacausal.ComponentCateEstimate
   metacausal.infer_outcome_type
```

### Warnings

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.MetaCausalWarning
   metacausal.ComponentWarning
   metacausal.ComponentFailureWarning
   metacausal.ComponentExclusionWarning
   metacausal.BootstrapWarning
```

## `metacausal.aggregation`

Aggregation strategies (combine component ATE/CATE predictions into an
ensemble estimate) and the result types they produce.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.aggregation.AggregationStrategy
   metacausal.aggregation.PointwiseStrategy
   metacausal.aggregation.AgreementStrategy
   metacausal.aggregation.SupervisedStrategy
   metacausal.aggregation.Mean
   metacausal.aggregation.Median
   metacausal.aggregation.TrimmedMean
   metacausal.aggregation.CBA
   metacausal.aggregation.CausalStacking
   metacausal.aggregation.QAggregation
   metacausal.aggregation.RStacking
   metacausal.aggregation.Select
   metacausal.aggregation.EnsembleWeights
   metacausal.aggregation.BootstrapResult
```

### Cross-fitting

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.aggregation.CrossFitSplit
   metacausal.aggregation.TrainAvgSplit
   metacausal.aggregation.FoldSpec
```

### Nuisance estimation

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.aggregation.NuisanceEstimates
   metacausal.aggregation.fit_nuisance
   metacausal.aggregation.dr_pseudo_outcome
   metacausal.aggregation.robinson_residuals
```

## `metacausal.adapters`

Wrappers presenting each supported library's estimators (EconML,
CausalML, DoubleML, stochtree) through one common interface, plus
generic adapters for wrapping arbitrary user-supplied estimators.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.adapters.CausalEstimator
   metacausal.adapters.EconMLAdapter
   metacausal.adapters.CausalMLAdapter
   metacausal.adapters.DoubleMLAdapter
   metacausal.adapters.StochtreeAdapter
   metacausal.adapters.GenericATEAdapter
   metacausal.adapters.GenericCATEAdapter
   metacausal.adapters.INNER_WORKER_ENV
```

## `metacausal.plots`

Requires the `plots` extra (`pip install "metacausal[plots]"`). Each
function is also available as a thin method on the class it plots --
see the corresponding class above (e.g. `BootstrapResult.forest()`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.plots.forest
   metacausal.plots.weights
   metacausal.plots.cate_profile
   metacausal.plots.disagreement
```

## `metacausal.datasets`

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   metacausal.datasets.load_lalonde
```
