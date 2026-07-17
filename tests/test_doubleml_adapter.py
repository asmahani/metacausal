"""Tests for DoubleMLAdapter — focus on reproducibility (#16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from metacausal.adapters.doubleml import DoubleMLAdapter

from doubleml import DoubleMLIRM, DoubleMLPLR


def _dgp(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, size=n)
    Y = X[:, 0] + T * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, T, Y


def _fast_r():
    return HistGradientBoostingRegressor(max_iter=20)


def _fast_c():
    return HistGradientBoostingClassifier(max_iter=20)


@pytest.mark.parametrize("cls,kwargs", [
    (DoubleMLIRM, {"ml_g": _fast_r(), "ml_m": _fast_c()}),
    (DoubleMLPLR, {"ml_l": _fast_r(), "ml_m": _fast_r()}),
], ids=["IRM", "PLR"])
def test_fit_reproducible_across_fresh_instances(cls, kwargs):
    """Regression for #16: DoubleML's sample splitting uses numpy's
    global RNG and exposes no seed. The adapter now snapshots and
    seeds ``np.random`` around the DoubleML construction + fit so
    two fresh adapter instances with the same random_state produce
    identical estimates."""
    X, T, Y = _dgp(n=200)
    a1 = DoubleMLAdapter(cls, n_folds=3, **kwargs)
    a1.fit(X, T, Y, random_state=42)
    ate1 = a1.ate().ate

    a2 = DoubleMLAdapter(cls, n_folds=3, **kwargs)
    a2.fit(X, T, Y, random_state=42)
    ate2 = a2.ate().ate

    assert ate1 == ate2


def test_fit_does_not_leak_np_random_state():
    """The seed/restore wrapper must leave ``np.random`` state intact
    so CausalEnsemble and user code outside the adapter see no side
    effects from the internal deterministic seeding."""
    X, T, Y = _dgp(n=200)
    np.random.seed(777)  # establish a known global state
    state_before = np.random.get_state()

    a = DoubleMLAdapter(
        DoubleMLIRM, ml_g=_fast_r(), ml_m=_fast_c(), n_folds=3,
    )
    a.fit(X, T, Y, random_state=42)

    state_after = np.random.get_state()
    # Compare the two state tuples element-by-element (state[1] is the
    # 624-uint32 key array; the rest are scalars).
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    for i in range(2, len(state_before)):
        assert state_before[i] == state_after[i]


def test_ate_forwards_alpha_to_confint_level():
    X, T, Y = _dgp(n=200)
    a = DoubleMLAdapter(
        DoubleMLIRM,
        alpha=0.10,
        ml_g=_fast_r(),
        ml_m=_fast_c(),
        n_folds=3,
    )
    a.fit(X, T, Y, random_state=42)

    calls = {}

    def fake_confint(*, level=0.95, joint=False):
        calls["level"] = level
        calls["joint"] = joint
        return pd.DataFrame([[-0.1, 0.2]], columns=["2.5 %", "97.5 %"])

    a._fitted_model.confint = fake_confint
    result = a.ate()

    assert result.ci_lower == -0.1
    assert result.ci_upper == 0.2
    assert calls == {"level": 0.9, "joint": False}


def test_importable_from_top_level():
    from metacausal import DoubleMLAdapter  # noqa: F401
