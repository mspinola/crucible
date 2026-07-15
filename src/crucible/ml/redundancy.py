"""crucible.ml.redundancy — which features say the same thing, and which one to keep.

Turns "I think some of these features overlap" into a ranked, evidence-backed
drop-list. Three ingredients:

  1. Group redundant features. Continuous features are grouped by |Spearman| (rank
     correlation); binary features by Cramér's V. In both cases a group is a
     connected component of the "association >= threshold" graph.
  2. Score each feature's relevance with `fold_ic` — its rank IC against the
     target, measured out-of-fold across chronological splits.
  3. For each redundant group, keep the member with the highest |IC| and drop the
     rest.

numpy/pandas only. The correlation is rank-Pearson (Spearman without scipy); the
grouping is graph connected-components rather than a hierarchical-linkage cut —
the standard "redundant if pairwise-associated above X" semantics, and the same
rule the binary side has always used.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RedundancyReport:
    """The evidence and the verdict.

    ``ic`` is the per-feature out-of-fold IC table (see :func:`fold_ic`).
    ``droplist`` has one row per feature in a redundant group, with a KEEP/DROP
    ``action`` (the highest-|IC| member of each group is kept).
    """

    ic: pd.DataFrame
    droplist: pd.DataFrame

    @property
    def dropped(self) -> list:
        """Features flagged DROP — redundant with a stronger sibling."""
        if self.droplist.empty:
            return []
        return self.droplist.loc[self.droplist["action"] == "DROP", "feature"].tolist()

    @property
    def kept(self) -> list:
        """The representative kept from each redundant group."""
        if self.droplist.empty:
            return []
        return self.droplist.loc[self.droplist["action"] == "KEEP", "feature"].tolist()


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V association between two categorical/binary series (0 = independent,
    1 = one determines the other). Returns 0.0 if either series is constant."""
    tbl = pd.crosstab(a, b)
    if tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return 0.0
    obs = tbl.to_numpy()
    n = obs.sum()
    expected = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0).sum()
    r, k = obs.shape
    return float(np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1)))


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman IC = Pearson of ranks, over pairwise non-null rows."""
    sa, sb = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    m = sa.notna() & sb.notna()
    if m.sum() < 2:
        return np.nan
    ic = sa[m].rank().corr(sb[m].rank())
    return float(ic) if pd.notna(ic) else np.nan


def _components(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components (size >= 2) of the graph defined by ``edges``, via
    union-find. Nodes not in any edge are not redundant and are omitted."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)

    comp: dict[str, list[str]] = {}
    for node in parent:
        comp.setdefault(find(node), []).append(node)
    return [sorted(m) for m in comp.values() if len(m) >= 2]


def _split_binary_continuous(panel: pd.DataFrame, features: list[str],
                             max_card: int) -> tuple[list[str], list[str]]:
    """A feature is 'binary' if it has <= ``max_card`` distinct non-null values."""
    binary, continuous = [], []
    for c in features:
        (binary if panel[c].dropna().nunique() <= max_card else continuous).append(c)
    return binary, continuous


def fold_ic(panel: pd.DataFrame, features: list[str], *, target: str,
            group: str | None = None, n_splits: int = 5, min_test: int = 10,
            min_group: int = 40) -> pd.DataFrame:
    """Per-feature out-of-fold rank IC against ``target``.

    Splits each ``group`` (e.g. a symbol; the whole frame if ``group`` is None)
    into ``n_splits`` chronological folds and measures each feature's Spearman IC
    against the target *within* each fold, then averages across all folds and
    groups. Rows are used in existing order, so pre-sort by time per group.

    Returns a table sorted by ``ic_abs``: ``feature``, ``ic_mean``, ``ic_abs``,
    ``ic_std``, ``ic_ir`` (mean/std info ratio), and ``n_folds``.
    """
    recs: dict[str, list[float]] = {f: [] for f in features}
    groups = [(None, panel)] if group is None else panel.groupby(group)
    for _, g in groups:
        n = len(g)
        if n < min_group:
            continue
        bounds = np.linspace(0, n, n_splits + 1).astype(int)
        y = g[target].to_numpy()
        for k in range(n_splits):
            test = np.arange(bounds[k], bounds[k + 1])
            if len(test) < min_test:
                continue
            yv = y[test]
            for f in features:
                fv = g[f].to_numpy()[test]
                if np.unique(fv[~pd.isna(fv)]).size < 2:
                    continue
                ic = _rank_ic(fv, yv)
                if not np.isnan(ic):
                    recs[f].append(ic)

    rows = [{
        "feature": f,
        "ic_mean": float(np.mean(v)) if v else np.nan,
        "ic_abs": abs(float(np.mean(v))) if v else np.nan,
        "ic_std": float(np.std(v)) if v else np.nan,
        "n_folds": len(v),
    } for f, v in recs.items()]
    out = pd.DataFrame(rows)
    out["ic_ir"] = out["ic_mean"] / out["ic_std"].replace(0, np.nan)
    return out.sort_values("ic_abs", ascending=False, na_position="last").reset_index(drop=True)


def redundancy_droplist(panel: pd.DataFrame, features: list[str], *, target: str,
                        group: str | None = None, corr_thresh: float = 0.85,
                        v_thresh: float = 0.80, binary_max_card: int = 3,
                        n_splits: int = 5) -> RedundancyReport:
    """Group redundant features, score them out-of-fold, and keep the best of each.

    Continuous features are grouped where |Spearman| >= ``corr_thresh``; binary
    features where Cramér's V >= ``v_thresh``. Within each redundant group the
    highest-|IC| feature is kept and the rest flagged DROP.
    """
    binary, continuous = _split_binary_continuous(panel, features, binary_max_card)
    ic = fold_ic(panel, features, target=target, group=group, n_splits=n_splits)
    ic_abs = ic.set_index("feature")["ic_abs"].to_dict()

    edges: list[tuple[str, str]] = []
    if len(continuous) >= 2:
        rho = panel[continuous].rank().corr().abs()
        edges += [(a, b) for a, b in combinations(continuous, 2)
                  if rho.loc[a, b] >= corr_thresh]
    cont_groups = _components(edges)

    bin_edges = [(a, b) for a, b in combinations(binary, 2)
                 if cramers_v(panel[a].fillna(0), panel[b].fillna(0)) >= v_thresh]
    bin_groups = _components(bin_edges)

    def keep_best(members: list[str]) -> str:
        return max(members, key=lambda m: -1.0 if pd.isna(ic_abs.get(m, np.nan))
                   else ic_abs[m])

    rows = []
    for kind, comps in (("cont", cont_groups), ("bin", bin_groups)):
        for cid, members in enumerate(comps):
            keep = keep_best(members)
            for m in members:
                v = ic_abs.get(m, np.nan)
                rows.append({"group": f"{kind}_{cid}", "feature": m,
                             "ic_abs": None if pd.isna(v) else round(v, 4),
                             "action": "KEEP" if m == keep else "DROP"})
    droplist = pd.DataFrame(rows, columns=["group", "feature", "ic_abs", "action"])
    return RedundancyReport(ic=ic, droplist=droplist)
