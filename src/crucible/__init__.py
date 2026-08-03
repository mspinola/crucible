"""crucible — run a trading strategy through the crucible; weak edges don't survive.

The capital-free core lives in :mod:`crucible.edge`: a trade log goes in, an edge
verdict — with a confidence interval and a p-value — comes out. No account, no
position sizing, no equity curve.
"""
from crucible import breadth, edge, ml, strategies, validation

# Kept in step with pyproject's `version` by a packaging-twin test in
# tests/test_boundaries.py — bump both together when cutting a release.
__version__ = "0.6.0"
__all__ = ["breadth", "edge", "ml", "strategies", "validation", "__version__"]
