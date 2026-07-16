"""Pointwise aggregation strategies: Median, Mean, and TrimmedMean."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metacausal.aggregation.base import PointwiseStrategy


@dataclass
class Median(PointwiseStrategy):
    """Pointwise median aggregation.

    Default aggregation strategy. 50% breakdown point: the ensemble is
    unaffected by up to half the component models producing wildly wrong
    estimates.

    Examples
    --------
    Four component CATE predictions for two evaluation points; the first
    point has one wild outlier (90.0) that the median ignores:

    >>> import numpy as np
    >>> from metacausal.aggregation import Median
    >>> values = np.array([[10.0, 5.0], [11.0, 5.0], [12.0, 5.0], [90.0, 5.0]])
    >>> Median().aggregate(values)
    array([11.5,  5. ])
    """

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        return np.median(values, axis=0)


@dataclass
class Mean(PointwiseStrategy):
    """Pointwise mean aggregation.

    Examples
    --------
    Same four component predictions as :class:`Median`'s example; here the
    outlier (90.0) drags the mean well above the other three models' ~11:

    >>> import numpy as np
    >>> from metacausal.aggregation import Mean
    >>> values = np.array([[10.0, 5.0], [11.0, 5.0], [12.0, 5.0], [90.0, 5.0]])
    >>> Mean().aggregate(values)
    array([30.75,  5.  ])
    """

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        return np.mean(values, axis=0)


@dataclass
class TrimmedMean(PointwiseStrategy):
    """Pointwise trimmed-mean aggregation.

    Drops the ``trim_count`` highest and lowest component estimates at each
    point, then averages the remainder. A tunable middle ground between
    :class:`Mean` (no trimming) and :class:`Median` (maximal trimming).

    Parameters
    ----------
    trim_count : int
        Number of models to drop from each tail. Default 1.
        Must satisfy ``2 * trim_count < K`` where K is the number of
        component models.

    Examples
    --------
    Same four component predictions as :class:`Median`'s example;
    ``trim_count=1`` drops the 90.0 outlier (and the lowest value, 10.0)
    before averaging, recovering the same robust result as the median:

    >>> import numpy as np
    >>> from metacausal.aggregation import TrimmedMean
    >>> values = np.array([[10.0, 5.0], [11.0, 5.0], [12.0, 5.0], [90.0, 5.0]])
    >>> TrimmedMean(trim_count=1).aggregate(values)
    array([11.5,  5. ])
    """

    trim_count: int = 1

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        k = values.shape[0]
        if 2 * self.trim_count >= k:
            raise ValueError(
                f"trim_count={self.trim_count} leaves no models after trimming "
                f"(K={k}). Reduce trim_count or add more component estimators."
            )
        sorted_arr = np.sort(values, axis=0)
        return sorted_arr[self.trim_count : k - self.trim_count].mean(axis=0)


# Maps string aliases to strategy classes. Values are classes, not instances —
# CausalEnsemble.__init__ calls them as constructors to get fresh instances.
_STRING_FACTORIES: dict[str, type[PointwiseStrategy]] = {
    "median": Median,
    "mean": Mean,
    "trimmed_mean": TrimmedMean,
}
