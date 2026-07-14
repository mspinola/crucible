"""crucible — run a trading strategy through the crucible; weak edges don't survive.

The capital-free core lives in :mod:`crucible.edge`: a trade log goes in, an edge
verdict — with a confidence interval and a p-value — comes out. No account, no
position sizing, no equity curve.
"""
from crucible import edge, strategies

__version__ = "0.1.0"
__all__ = ["edge", "strategies", "__version__"]
