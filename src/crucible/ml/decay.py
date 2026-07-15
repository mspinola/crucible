"""crucible.ml.decay — does a higher score mean a better outcome?

Bucket a score into equal-count quantiles and read the realized win rate per
bucket. A genuine, well-ordered edge makes win rate climb monotonically from the
worst quantile to the best; a flat or ragged profile is the tell of a score that
ranks nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecayTable:
    """Per-quantile outcome of a score.

    ``table`` has one row per quantile with ``win_rate`` / ``avg_score`` /
    ``count``; ``monotonic`` is True when win rate is non-decreasing across every
    quantile (the signature of a real, well-ordered edge).
    """

    table: pd.DataFrame
    monotonic: bool
    q: int

    @property
    def spread(self) -> float:
        """Top-quantile win rate minus bottom — the crude 'does it separate' number."""
        return float(self.table["win_rate"].iloc[-1] - self.table["win_rate"].iloc[0])


def quantile_decay(preds: pd.DataFrame, *, score: str = "score",
                   label: str = "label", q: int = 5) -> DecayTable:
    """Bucket ``score`` into ``q`` equal-count quantiles and report, per bucket,
    the realized win rate (fraction with ``label > 0``), the average score, and the
    count. A genuine edge makes win rate rise monotonically from Q1 to Q``q``.

    ``win`` is defined as ``label > 0``, so it reads correctly whether losers are
    encoded as ``-1`` or ``0`` — the ambiguity that made the original 'alphalens'
    tearsheet fall back mid-computation. Scores are ranked before bucketing so
    exact-duplicate scores don't collapse a quantile.
    """
    if score not in preds.columns or label not in preds.columns:
        raise ValueError(f"preds needs '{score}' and '{label}' columns")
    df = preds[[score, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < q:
        raise ValueError(f"need at least q={q} valid rows, got {len(df)}")

    df = df.assign(
        _win=(df[label] > 0).astype(float),
        _q=pd.qcut(df[score].rank(method="first"), q, labels=range(1, q + 1)),
    )
    table = (df.groupby("_q", observed=True)
               .agg(win_rate=("_win", "mean"),
                    avg_score=(score, "mean"),
                    count=("_win", "size"))
               .reset_index()
               .rename(columns={"_q": "quantile"}))
    monotonic = bool(np.all(np.diff(table["win_rate"].to_numpy()) >= 0))
    return DecayTable(table=table, monotonic=monotonic, q=q)
