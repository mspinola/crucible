"""crucible.breadth — how many *independent* bets a set of return streams holds.

Correlated markets aren't independent evidence: 8 currency futures move as ~1
dollar bet. ``N_eff`` is the participation ratio of the return-correlation
eigenvalues, (Σλ)² / Σλ² — it equals N for perfectly independent streams and 1
for perfectly correlated ones. It's the honest denominator when you ask whether
an edge measured across a book is real: a 665-trade book spread over ~14
independent market-bets carries less evidence than the raw count implies.

Capital-free, like the rest of crucible: correlation structure only — no sizing,
no equity curve. (The drawdown *consequence* of that structure needs a capital
model, which is out of scope here.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Breadth:
    """The independence structure of a panel of return streams.

    ``n_eff`` is the effective number of independent bets; ``loadings`` (assets ×
    PC1..PCn, descending by variance explained) name the factors so a caller can
    label them (dollar / rates / grains / …) without re-running the decomposition.
    """

    n_eff: float                 # effective number of independent bets
    n_assets: int                # raw count of streams
    eigenvalues: np.ndarray      # correlation eigenvalues, descending
    loadings: pd.DataFrame       # assets × PCs (PC1..PCn), descending by eigenvalue
    corr: pd.DataFrame           # the correlation matrix it came from

    @property
    def redundancy(self) -> float:
        """n_assets / n_eff — streams carried per independent bet."""
        return self.n_assets / self.n_eff


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """(Σλ)² / Σλ² — N_eff from a set of correlation eigenvalues.

    For an N×N correlation matrix the eigenvalues sum to N, so this is bounded in
    [1, N]: 1 when one factor explains everything, N when all N are independent.
    """
    w = np.asarray(eigenvalues, dtype=float)
    denom = (w ** 2).sum()
    if denom == 0:
        raise ValueError("eigenvalues sum to zero — no variance to decompose")
    return float(w.sum() ** 2 / denom)


def effective_n(returns: pd.DataFrame, min_obs: int = 30) -> Breadth:
    """Effective number of independent bets in a panel of return streams.

    ``returns`` is a (dates × assets) frame of aligned periodic returns — weekly,
    daily, whatever, as long as the columns share a clock. Streams with fewer than
    ``min_obs`` observations are dropped first (so a late-starting asset doesn't
    shrink everyone's window), then the remaining columns are reduced to their
    complete-case rows for a clean positive-semidefinite correlation matrix.
    Nothing here is capital-aware; pass raw return streams.

    Raises ``ValueError`` if fewer than two streams survive with overlapping
    history.
    """
    R = returns.replace([np.inf, -np.inf], np.nan)
    R = R.loc[:, R.notna().sum() >= min_obs]   # drop short-history streams first
    R = R.dropna()                             # then complete-case rows
    if R.shape[1] < 2 or R.empty:
        raise ValueError("need >= 2 return streams with overlapping history")

    corr = R.corr()
    w, V = np.linalg.eigh(corr.values)         # symmetric -> eigh; eigenvalues ascending
    order = np.argsort(w)[::-1]                # descending by variance explained
    w, V = w[order], V[:, order]
    loadings = pd.DataFrame(
        V, index=corr.columns,
        columns=[f"PC{i + 1}" for i in range(len(w))],
    )
    return Breadth(
        n_eff=participation_ratio(w),
        n_assets=corr.shape[0],
        eigenvalues=w,
        loadings=loadings,
        corr=corr,
    )
