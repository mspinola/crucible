"""crucible.ml — is an ML *signal* real, or leaked / redundant / overfit?

The honesty layer applied to a model's scores instead of a trade log. Model-
agnostic and capital-free: the core imports only numpy/pandas (no sklearn/xgboost
— those belong to whoever fits the model). A *predictions* frame carries a
continuous ``score`` and the realized ``label`` (+1 winner / -1 or 0 loser) it
tried to call.

Landed so far:

  information_coefficient / alpha_gate   how much predictive signal a score holds
  quantile_decay                         does a higher score mean a better outcome?

Follow-ons (still in pardo): purged cross-validated IC, feature-redundancy
droplists, point-in-time windowing, and the Plotly decay tearsheet.
"""
from crucible.ml.ic import AlphaGateError, alpha_gate, information_coefficient
from crucible.ml.decay import DecayTable, quantile_decay

__all__ = [
    "information_coefficient", "alpha_gate", "AlphaGateError",
    "quantile_decay", "DecayTable",
]
