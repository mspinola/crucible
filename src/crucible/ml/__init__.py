"""crucible.ml — is an ML *signal* real, or leaked / redundant / overfit?

The honesty layer applied to a model's scores instead of a trade log. Model-
agnostic and capital-free: the core imports only numpy/pandas (no sklearn/xgboost
— those belong to whoever fits the model). A *predictions* frame carries a
continuous ``score`` and the realized ``label`` (+1 winner / -1 or 0 loser) it
tried to call.

Landed so far:

  information_coefficient / alpha_gate   how much predictive signal a score holds
  quantile_decay / decay_tearsheet       does a higher score mean a better outcome?
  score_by_outcome                       winners-vs-losers score violins (a panel)
  fold_ic                                out-of-fold rank IC per feature
  redundancy_droplist / cramers_v        which features overlap, and which to keep
  asof_window / window_before            point-in-time slices that can't peek ahead

``decay_tearsheet`` needs plotly (the ``report`` extra); everything else is
numpy/pandas only.
"""
from crucible.ml.ic import AlphaGateError, alpha_gate, information_coefficient
from crucible.ml.decay import DecayTable, decay_tearsheet, quantile_decay, score_by_outcome
from crucible.ml.redundancy import (
    RedundancyReport, cramers_v, fold_ic, redundancy_droplist,
)
from crucible.ml.pit import asof_window, window_before

__all__ = [
    "information_coefficient", "alpha_gate", "AlphaGateError",
    "quantile_decay", "decay_tearsheet", "score_by_outcome", "DecayTable",
    "fold_ic", "redundancy_droplist", "cramers_v", "RedundancyReport",
    "asof_window", "window_before",
]
