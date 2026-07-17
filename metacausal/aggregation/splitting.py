"""Data splitting strategies for supervised CATE ensemble aggregation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


# ---------------------------------------------------------------------------
# FoldSpec
# ---------------------------------------------------------------------------


@dataclass
class FoldSpec:
    """Result of a data split: absolute indices into the original array per fold.

    Attributes
    ----------
    train_indices : list of arrays
        train_indices[j] contains the row indices used for training in fold j.
    test_indices : list of arrays
        test_indices[j] contains the row indices held out for evaluation in fold j.
        For CrossFitSplit these are out-of-fold (OOF) indices.
        For TrainAvgSplit this is the "averaging set" where weights are optimized.
    n_folds : int
        Number of folds. Equals len(train_indices) == len(test_indices).
    """

    train_indices: list[np.ndarray]
    test_indices: list[np.ndarray]
    n_folds: int

    def __post_init__(self) -> None:
        if len(self.train_indices) != self.n_folds:
            raise ValueError(
                f"n_folds={self.n_folds} does not match "
                f"len(train_indices)={len(self.train_indices)}"
            )
        if len(self.test_indices) != self.n_folds:
            raise ValueError(
                f"n_folds={self.n_folds} does not match "
                f"len(test_indices)={len(self.test_indices)}"
            )


# ---------------------------------------------------------------------------
# CrossFitSplit
# ---------------------------------------------------------------------------


@dataclass
class CrossFitSplit:
    """Q-fold cross-fitting split.

    Every observation gets exactly one out-of-fold (OOF) prediction.
    Wraps sklearn's StratifiedKFold (when stratify=True) or KFold.

    Parameters
    ----------
    n_folds : int
        Number of folds (Q). Must be >= 2.
    stratify : bool
        If True, stratify folds on treatment assignment T. Requires T to be
        discrete (binary or categorical). If T appears continuous (more than
        10 unique values), a warning is issued and stratification is skipped.

    Examples
    --------
    >>> import numpy as np
    >>> from metacausal.aggregation import CrossFitSplit
    >>> T = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    >>> fold_spec = CrossFitSplit(n_folds=5).split(T, random_state=0)
    >>> fold_spec.n_folds
    5
    >>> np.sort(np.concatenate(fold_spec.test_indices))  # every index appears once
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """

    n_folds: int = 5
    stratify: bool = True

    def split(self, T: np.ndarray, random_state: int | None = None) -> FoldSpec:
        """Partition n observations into Q folds.

        Parameters
        ----------
        T : array of shape (n,)
            Treatment assignments. Used for stratification when stratify=True.
        random_state : int or None
            Random seed for reproducibility. Lives on the call, not the object,
            so the same CrossFitSplit can be reused with different seeds.

        Returns
        -------
        FoldSpec with n_folds folds. Every index in range(n) appears in
        exactly one test fold.
        """
        T = np.asarray(T)
        n = len(T)

        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.n_folds > n:
            raise ValueError(
                f"n_folds={self.n_folds} cannot exceed n={n}"
            )

        use_stratify = self._check_stratify(T)

        if use_stratify:
            splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=random_state
            )
            splits = list(splitter.split(np.zeros(n), T))
        else:
            splitter = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=random_state
            )
            splits = list(splitter.split(np.zeros(n)))

        train_indices = [train for train, _ in splits]
        test_indices = [test for _, test in splits]

        return FoldSpec(
            train_indices=train_indices,
            test_indices=test_indices,
            n_folds=self.n_folds,
        )

    def _check_stratify(self, T: np.ndarray) -> bool:
        """Return whether to actually stratify, warning if T looks continuous."""
        if not self.stratify:
            return False
        n_unique = len(np.unique(T))
        if n_unique > 10:
            warnings.warn(
                f"stratify=True but T has {n_unique} unique values, which suggests "
                "continuous treatment. Stratification requires discrete T. "
                "Falling back to unstratified KFold.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True


# ---------------------------------------------------------------------------
# TrainAvgSplit
# ---------------------------------------------------------------------------


@dataclass
class TrainAvgSplit:
    """Simple train / averaging-set split.

    Holds out a fraction of observations as the "averaging set" where ensemble
    weights are optimized. Component models are trained on the remainder.

    Parameters
    ----------
    avg_frac : float
        Fraction of observations held out for weight optimization.
        Must be in (0, 1). Typical values: 0.2–0.3.
    stratify : bool
        If True, stratify the split on treatment assignment T. Subject to the
        same discrete-T requirement as CrossFitSplit.

    Examples
    --------
    >>> import numpy as np
    >>> from metacausal.aggregation import TrainAvgSplit
    >>> T = np.array([0, 1] * 10)
    >>> fold_spec = TrainAvgSplit(avg_frac=0.3).split(T, random_state=0)
    >>> fold_spec.n_folds
    1
    >>> len(fold_spec.train_indices[0]), len(fold_spec.test_indices[0])
    (14, 6)
    """

    avg_frac: float = 0.25
    stratify: bool = True

    def split(self, T: np.ndarray, random_state: int | None = None) -> FoldSpec:
        """Partition n observations into training and averaging sets.

        Parameters
        ----------
        T : array of shape (n,)
            Treatment assignments. Used for stratification when stratify=True.
        random_state : int or None
            Random seed for reproducibility.

        Returns
        -------
        FoldSpec with n_folds=1.
            train_indices[0] — indices for fitting component models.
            test_indices[0]  — indices for optimizing ensemble weights.
        """
        T = np.asarray(T)
        n = len(T)

        if not (0.0 < self.avg_frac < 1.0):
            raise ValueError(
                f"avg_frac must be in (0, 1), got {self.avg_frac}"
            )

        min_avg = 2  # need at least 2 observations for weight optimization
        if int(n * self.avg_frac) < min_avg:
            raise ValueError(
                f"avg_frac={self.avg_frac} with n={n} yields fewer than "
                f"{min_avg} averaging-set observations"
            )

        use_stratify = self._check_stratify(T)
        stratify_arg = T if use_stratify else None

        indices = np.arange(n)
        train_idx, avg_idx = train_test_split(
            indices,
            test_size=self.avg_frac,
            stratify=stratify_arg,
            random_state=random_state,
        )

        return FoldSpec(
            train_indices=[train_idx],
            test_indices=[avg_idx],
            n_folds=1,
        )

    def _check_stratify(self, T: np.ndarray) -> bool:
        """Return whether to actually stratify, warning if T looks continuous."""
        if not self.stratify:
            return False
        n_unique = len(np.unique(T))
        if n_unique > 10:
            warnings.warn(
                f"stratify=True but T has {n_unique} unique values, which suggests "
                "continuous treatment. Stratification requires discrete T. "
                "Falling back to unstratified split.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True
