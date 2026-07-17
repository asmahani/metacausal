"""Tests for CrossFitSplit and TrainAvgSplit."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from metacausal.aggregation.splitting import CrossFitSplit, FoldSpec, TrainAvgSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_T(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n).astype(float)


def _all_test_indices(fold_spec: FoldSpec) -> np.ndarray:
    """Concatenate all test indices across folds."""
    return np.concatenate(fold_spec.test_indices)


# ---------------------------------------------------------------------------
# FoldSpec validation
# ---------------------------------------------------------------------------


class TestFoldSpec:
    def test_valid_construction(self):
        train = [np.array([0, 1]), np.array([2, 3])]
        test = [np.array([2, 3]), np.array([0, 1])]
        fs = FoldSpec(train_indices=train, test_indices=test, n_folds=2)
        assert fs.n_folds == 2

    def test_mismatched_train_length_raises(self):
        train = [np.array([0, 1])]  # only 1 fold
        test = [np.array([2]), np.array([3])]
        with pytest.raises(ValueError, match="n_folds=2"):
            FoldSpec(train_indices=train, test_indices=test, n_folds=2)

    def test_mismatched_test_length_raises(self):
        train = [np.array([0, 1]), np.array([2, 3])]
        test = [np.array([2, 3])]  # only 1 fold
        with pytest.raises(ValueError, match="n_folds=2"):
            FoldSpec(train_indices=train, test_indices=test, n_folds=2)


# ---------------------------------------------------------------------------
# CrossFitSplit
# ---------------------------------------------------------------------------


class TestCrossFitSplit:
    def test_fold_coverage(self):
        """Every index appears in exactly one test fold."""
        n = 100
        T = _binary_T(n)
        fs = CrossFitSplit(n_folds=5).split(T, random_state=0)

        all_test = np.sort(_all_test_indices(fs))
        assert np.array_equal(all_test, np.arange(n)), "Test indices do not cover all n"

    def test_no_test_overlap(self):
        """Test folds are disjoint."""
        n = 100
        T = _binary_T(n)
        fs = CrossFitSplit(n_folds=5).split(T, random_state=0)

        seen = set()
        for idx in fs.test_indices:
            fold_set = set(idx.tolist())
            assert fold_set.isdisjoint(seen), "Test folds overlap"
            seen |= fold_set

    def test_n_folds_matches(self):
        T = _binary_T(50)
        fs = CrossFitSplit(n_folds=4).split(T, random_state=1)
        assert fs.n_folds == 4
        assert len(fs.train_indices) == 4
        assert len(fs.test_indices) == 4

    def test_reproducibility(self):
        """Same seed produces identical fold assignments."""
        T = _binary_T(80)
        fs1 = CrossFitSplit(n_folds=5).split(T, random_state=42)
        fs2 = CrossFitSplit(n_folds=5).split(T, random_state=42)
        for a, b in zip(fs1.test_indices, fs2.test_indices):
            assert np.array_equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds generally produce different assignments."""
        T = _binary_T(80)
        fs1 = CrossFitSplit(n_folds=5).split(T, random_state=0)
        fs2 = CrossFitSplit(n_folds=5).split(T, random_state=99)
        # With n=80 and different seeds it would be astronomically unlikely
        # for all folds to be identical
        any_different = any(
            not np.array_equal(a, b)
            for a, b in zip(fs1.test_indices, fs2.test_indices)
        )
        assert any_different

    def test_stratification_balances_treatment(self):
        """Stratified folds have approximately equal treatment proportions."""
        rng = np.random.default_rng(0)
        n = 200
        T = rng.integers(0, 2, size=n).astype(float)
        overall_rate = T.mean()

        fs = CrossFitSplit(n_folds=5, stratify=True).split(T, random_state=0)

        for test_idx in fs.test_indices:
            fold_rate = T[test_idx].mean()
            assert abs(fold_rate - overall_rate) < 0.15, (
                f"Fold treatment rate {fold_rate:.2f} deviates from "
                f"overall {overall_rate:.2f}"
            )

    def test_unstratified_runs(self):
        """stratify=False runs without error."""
        T = _binary_T(60)
        fs = CrossFitSplit(n_folds=3, stratify=False).split(T, random_state=7)
        assert fs.n_folds == 3

    def test_continuous_T_warns_and_falls_back(self):
        """Continuous T with stratify=True emits a warning and uses KFold."""
        rng = np.random.default_rng(0)
        T = rng.uniform(0, 1, size=100)  # continuous
        splitter = CrossFitSplit(n_folds=5, stratify=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fs = splitter.split(T, random_state=0)
        assert any("continuous" in str(w.message).lower() for w in caught)
        # Should still produce a valid split
        assert fs.n_folds == 5
        all_test = np.sort(_all_test_indices(fs))
        assert np.array_equal(all_test, np.arange(100))

    def test_n_folds_less_than_2_raises(self):
        T = _binary_T(20)
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            CrossFitSplit(n_folds=1).split(T)

    def test_n_folds_exceeds_n_raises(self):
        T = _binary_T(5)
        with pytest.raises(ValueError, match="cannot exceed n"):
            CrossFitSplit(n_folds=10).split(T)

    def test_train_size_approximately_correct(self):
        """Each training fold contains ~(Q-1)/Q * n observations."""
        n, Q = 100, 5
        T = _binary_T(n)
        fs = CrossFitSplit(n_folds=Q).split(T, random_state=0)
        expected_train = n * (Q - 1) / Q
        for train_idx in fs.train_indices:
            assert abs(len(train_idx) - expected_train) <= 2


# ---------------------------------------------------------------------------
# TrainAvgSplit
# ---------------------------------------------------------------------------


class TestTrainAvgSplit:
    def test_n_folds_is_one(self):
        T = _binary_T(100)
        fs = TrainAvgSplit(avg_frac=0.25).split(T, random_state=0)
        assert fs.n_folds == 1
        assert len(fs.train_indices) == 1
        assert len(fs.test_indices) == 1

    def test_coverage(self):
        """train and averaging sets together cover all n indices."""
        n = 100
        T = _binary_T(n)
        fs = TrainAvgSplit(avg_frac=0.25).split(T, random_state=0)
        all_idx = np.sort(
            np.concatenate([fs.train_indices[0], fs.test_indices[0]])
        )
        assert np.array_equal(all_idx, np.arange(n))

    def test_no_overlap(self):
        """Train and averaging sets are disjoint."""
        T = _binary_T(100)
        fs = TrainAvgSplit(avg_frac=0.25).split(T, random_state=0)
        train_set = set(fs.train_indices[0].tolist())
        avg_set = set(fs.test_indices[0].tolist())
        assert train_set.isdisjoint(avg_set)

    def test_avg_frac_size(self):
        """Averaging set size is approximately avg_frac * n."""
        n = 200
        T = _binary_T(n)
        avg_frac = 0.3
        fs = TrainAvgSplit(avg_frac=avg_frac).split(T, random_state=0)
        expected = int(n * avg_frac)
        actual = len(fs.test_indices[0])
        # train_test_split rounds; allow ±1
        assert abs(actual - expected) <= 1

    def test_reproducibility(self):
        T = _binary_T(80)
        fs1 = TrainAvgSplit(avg_frac=0.25).split(T, random_state=13)
        fs2 = TrainAvgSplit(avg_frac=0.25).split(T, random_state=13)
        assert np.array_equal(fs1.train_indices[0], fs2.train_indices[0])
        assert np.array_equal(fs1.test_indices[0], fs2.test_indices[0])

    def test_stratification_balances_treatment(self):
        rng = np.random.default_rng(0)
        n = 300
        T = rng.integers(0, 2, size=n).astype(float)
        overall_rate = T.mean()

        fs = TrainAvgSplit(avg_frac=0.25, stratify=True).split(T, random_state=0)
        avg_rate = T[fs.test_indices[0]].mean()
        assert abs(avg_rate - overall_rate) < 0.1

    def test_unstratified_runs(self):
        T = _binary_T(60)
        fs = TrainAvgSplit(avg_frac=0.2, stratify=False).split(T, random_state=0)
        assert fs.n_folds == 1

    def test_continuous_T_warns_and_falls_back(self):
        rng = np.random.default_rng(0)
        T = rng.uniform(0, 1, size=100)
        splitter = TrainAvgSplit(avg_frac=0.25, stratify=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fs = splitter.split(T, random_state=0)
        assert any("continuous" in str(w.message).lower() for w in caught)
        assert fs.n_folds == 1

    def test_avg_frac_zero_raises(self):
        T = _binary_T(50)
        with pytest.raises(ValueError, match="avg_frac must be in"):
            TrainAvgSplit(avg_frac=0.0).split(T)

    def test_avg_frac_one_raises(self):
        T = _binary_T(50)
        with pytest.raises(ValueError, match="avg_frac must be in"):
            TrainAvgSplit(avg_frac=1.0).split(T)

    def test_avg_frac_too_small_raises(self):
        """avg_frac that produces < 2 averaging observations raises."""
        T = _binary_T(10)
        with pytest.raises(ValueError, match="fewer than"):
            TrainAvgSplit(avg_frac=0.05).split(T)

# ---------------------------------------------------------------------------
# Import from package
# ---------------------------------------------------------------------------


def test_importable_from_aggregation_package():
    from metacausal.aggregation import CrossFitSplit, FoldSpec, TrainAvgSplit  # noqa: F401
