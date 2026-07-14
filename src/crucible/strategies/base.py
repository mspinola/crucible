"""Shared helpers for the example strategies."""
from __future__ import annotations

import pandas as pd


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where `fast` crosses strictly above `slow` (the event bar)."""
    prev = fast.shift(1) <= slow.shift(1)
    now = fast > slow
    return (prev & now).fillna(False)
