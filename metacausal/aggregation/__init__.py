"""Aggregation strategies for CausalEnsemble."""

from metacausal.aggregation.base import AggregationStrategy, AgreementStrategy, PointwiseStrategy, SupervisedStrategy
from metacausal.aggregation.causal_stacking import CausalStacking
from metacausal.aggregation.cba import CBA
from metacausal.aggregation.nuisance import NuisanceEstimates, dr_pseudo_outcome, fit_nuisance, robinson_residuals
from metacausal.aggregation.pointwise import Mean, Median, TrimmedMean, _STRING_FACTORIES as _POINTWISE_FACTORIES
from metacausal.aggregation.q_aggregation import QAggregation
from metacausal.aggregation.r_stacking import RStacking
from metacausal.aggregation.select import Select
from metacausal.aggregation.splitting import CrossFitSplit, FoldSpec, TrainAvgSplit
from metacausal.aggregation.weights import BootstrapResult, EnsembleWeights

# All string aliases, including agreement strategies.
# Values are classes (not instances); CausalEnsemble.__init__ calls them as constructors.
_STRING_FACTORIES = {
    **_POINTWISE_FACTORIES,
    "cba": CBA,
}

__all__ = [
    "AggregationStrategy",
    "PointwiseStrategy",
    "AgreementStrategy",
    "SupervisedStrategy",
    "Median",
    "Mean",
    "TrimmedMean",
    "CBA",
    "EnsembleWeights",
    "BootstrapResult",
    "FoldSpec",
    "CrossFitSplit",
    "TrainAvgSplit",
    "NuisanceEstimates",
    "fit_nuisance",
    "dr_pseudo_outcome",
    "robinson_residuals",
    "CausalStacking",
    "QAggregation",
    "RStacking",
    "Select",
]
