"""Built-in datasets for MetaCausal."""

from __future__ import annotations

import importlib.resources
from typing import Literal

import numpy as np
import pandas as pd


def load_lalonde(
    raw: bool = False,
    binarize_y: Literal[None, "median", "positive"] = None,
) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the Lalonde job training dataset.

    The dataset contains 445 observations with a binary treatment indicator
    (job training program) and a continuous outcome (earnings in 1978).

    Parameters:
        raw: If True, return a DataFrame. If False (default), return
            ``(X, T, Y)`` numpy arrays.
        binarize_y: Optional binarization of the 1978 earnings outcome.
            ``None`` (default) keeps it continuous. ``"median"`` thresholds
            at the sample median of ``re78`` (~50/50 split, useful as a
            balanced binary fixture). ``"positive"`` thresholds at
            ``re78 > 0`` (~69/31 split, the natural "any 1978 earnings"
            indicator). Ignored when ``raw=True``.

    Returns:
        If ``raw=False``: tuple of (X, T, Y) where

            - X: covariate matrix, shape (n, 10)
            - T: binary treatment vector, shape (n,)
            - Y: outcome vector, shape (n,) — continuous if
              ``binarize_y`` is ``None``, integer 0/1 otherwise.

        If ``raw=True``: DataFrame with all columns.

    Examples:
        >>> from metacausal.datasets import load_lalonde
        >>> X, T, Y = load_lalonde()
        >>> X.shape
        (445, 10)
        >>> sorted(set(T.tolist()))
        [0, 1]
        >>> X, T, Y_bin = load_lalonde(binarize_y="median")
        >>> sorted(set(Y_bin.tolist()))
        [0, 1]
    """
    pkg = importlib.resources.files("metacausal.datasets")
    path = pkg.joinpath("lalonde.csv")
    with path.open("r") as fh:
        df = pd.read_csv(fh)

    if raw:
        return df

    T = df["treat"].to_numpy(dtype=int)
    re78 = df["re78"].to_numpy(dtype=float)
    X = df.drop(columns=["treat", "re78"]).to_numpy(dtype=float)

    if binarize_y is None:
        Y = re78
    elif binarize_y == "median":
        Y = (re78 > np.median(re78)).astype(int)
    elif binarize_y == "positive":
        Y = (re78 > 0).astype(int)
    else:
        raise ValueError(
            f"binarize_y must be None, 'median', or 'positive'; "
            f"got {binarize_y!r}."
        )

    return X, T, Y
