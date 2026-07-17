"""Visualisation helpers for MetaCausal results.

This subpackage requires matplotlib, declared as the optional
``[plots]`` extra: ``pip install 'metacausal[plots]'``.

Functions
---------
forest
    Component and ensemble ATEs on a forest plot.
weights
    Aggregation weights as a bar chart.
cate_profile
    Ensemble CATE (with optional bootstrap CI) along one covariate.
disagreement
    Pairwise agreement of component CATEs as a heatmap.

Every function accepts an optional ``ax`` and returns the primary
``matplotlib.axes.Axes`` it drew on. Pass an existing ``ax`` to compose
multi-panel figures; leave it ``None`` to create a fresh figure.
"""

from __future__ import annotations

try:
    import matplotlib  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "metacausal.plots requires matplotlib. "
        "Install the plots extra: pip install 'metacausal[plots]'."
    ) from exc

from metacausal.plots.cate_profile import cate_profile
from metacausal.plots.disagreement import disagreement
from metacausal.plots.forest import forest
from metacausal.plots.weights import weights

__all__ = ["forest", "weights", "cate_profile", "disagreement"]
