"""MACD signal-line crossover — a second worked example."""
from __future__ import annotations

import pandas as pd

from crucible.strategies.base import crossover


def macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
               price: str = "Close") -> pd.Series:
    """Entry on each bullish MACD cross (MACD line crosses above its signal line).
    Returns a boolean Series aligned to `df`."""
    p = df[price]
    macd = p.ewm(span=fast, adjust=False).mean() - p.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return crossover(macd, sig)
