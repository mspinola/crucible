"""Moving-average crossover — the canonical example signal."""
from __future__ import annotations

import pandas as pd

from crucible.strategies.base import crossover


def ma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50,
             price: str = "Close", kind: str = "sma") -> pd.Series:
    """Entry on each bar where the fast MA crosses above the slow MA (long).

    kind: 'sma' (simple) or 'ema' (exponential). Returns a boolean Series aligned
    to `df`, ready to hand to `barrier_trades`.
    """
    p = df[price]
    if kind == "ema":
        f, s = p.ewm(span=fast, adjust=False).mean(), p.ewm(span=slow, adjust=False).mean()
    else:
        f, s = p.rolling(fast).mean(), p.rolling(slow).mean()
    return crossover(f, s)
