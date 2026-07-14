"""Example signal generators. These replace the private CMR/news-failure logic —
each returns a boolean entry Series aligned to the OHLC frame, ready for
:func:`crucible.edge.barrier_trades`. They exist so the README runs and to show
the shape of a signal; they are demos, not endorsed edges.
"""
from crucible.strategies.ma_cross import ma_cross
from crucible.strategies.macd_cross import macd_cross

__all__ = ["ma_cross", "macd_cross"]
