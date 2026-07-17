"""Tests pinning the ABC contract for the four aggregation-strategy bases.

Each base must reject direct instantiation with a TypeError that names
the missing abstract method(s). Concrete strategies that override the
abstracts must instantiate without error. These guards ensure that a
future strategy author cannot accidentally drop a required abstract
override and have the failure surface only at fit time.
"""

from __future__ import annotations

import numpy as np
import pytest

from metacausal.aggregation import (
    AgreementStrategy,
    AggregationStrategy,
    CBA,
    CausalStacking,
    Mean,
    Median,
    PointwiseStrategy,
    QAggregation,
    RStacking,
    SupervisedStrategy,
    TrimmedMean,
)


class TestAbstractBasesCannotInstantiate:
    """Direct instantiation of any abstract base must raise TypeError."""

    def test_aggregation_strategy(self):
        with pytest.raises(TypeError, match="abstract"):
            AggregationStrategy()

    def test_pointwise_strategy_lists_aggregate(self):
        with pytest.raises(TypeError, match=r"aggregate\b"):
            PointwiseStrategy()

    def test_agreement_strategy_lists_compute_weights(self):
        with pytest.raises(TypeError, match=r"compute_weights\b"):
            AgreementStrategy()

    def test_supervised_strategy_lists_fit_weights(self):
        with pytest.raises(TypeError, match=r"fit_weights\b"):
            SupervisedStrategy()


class TestConcreteStrategiesInstantiate:
    """All shipped concrete strategies must instantiate cleanly."""

    @pytest.mark.parametrize(
        "cls",
        [Median, Mean, TrimmedMean, CBA, CausalStacking, RStacking, QAggregation],
    )
    def test_no_args_default_construction(self, cls):
        cls()  # must not raise


class TestPartialOverrideStillAbstract:
    """A subclass that overrides only some abstracts stays abstract."""

    def test_pointwise_subclass_without_aggregate(self):
        class IncompletePointwise(PointwiseStrategy):
            pass

        with pytest.raises(TypeError, match=r"aggregate\b"):
            IncompletePointwise()

    def test_agreement_subclass_without_compute_weights(self):
        class IncompleteAgreement(AgreementStrategy):
            pass

        with pytest.raises(TypeError, match=r"compute_weights\b"):
            IncompleteAgreement()

    def test_supervised_subclass_without_fit_weights(self):
        class IncompleteSupervised(SupervisedStrategy):
            pass

        with pytest.raises(TypeError, match=r"fit_weights\b"):
            IncompleteSupervised()


class TestPointwiseExtendedContract:
    """PointwiseStrategy.aggregate accepts both 1-D and 2-D input."""

    def test_median_1d_returns_scalar(self):
        result = Median().aggregate(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(float(result), 2.0)

    def test_median_2d_returns_vector(self):
        result = Median().aggregate(np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
        np.testing.assert_allclose(result, [2.0, 5.0])
