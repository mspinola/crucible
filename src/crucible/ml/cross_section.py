"""crucible.ml.cross_section: does a *ranking* carry information?

A ranking strategy (relative momentum, any factor rotation) has no entry event and
no stop, so there is no natural R-multiple and no timing to randomize. The honest
questions are cross-sectional:

  rank_band_decay         cut each date's ranking into bands of ``band_size`` and
                          read return / win rate / payoff per band, every band on
                          the same dates. A real factor decays down the ranking;
                          *how* it decays (slope vs cliff) is itself a finding.
  random_selection_null   does the top-N beat what the same selection rule earns
                          when each name's score path is handed to a different
                          name? Market moves, a score's persistence and each
                          name's own drift all survive into the null, so none of
                          them can pass for skill.

Input is a long *panel*: one row per (date, name) with a ``score`` known at the
date and a ``label`` = the forward return realized over the holding period that
follows. Point-in-time membership and a survivorship-free universe are the
caller's job (CLEAN): crucible cannot see a name that the panel omits. A name that
IS in the panel with a score but no label (typically a delisting inside the
holding period) is refused by default rather than silently dropped, because
dropping it is exactly how survivorship bias gets in.

Both functions report raw, uncorrected statistics. Every ``top_n`` / ``band_size``
tried is a variant: record each in a ``SearchSpaceLog``.

numpy/pandas only, capital-free, deterministic given ``seed``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MISSING_LABEL = ("raise", "drop")


def _clean(panel: pd.DataFrame, cols: list[str], date: str, score: str, label: str,
           missing_label: str) -> tuple[pd.DataFrame, int]:
    """Drop unscored rows and unrealized dates; refuse (or count) scored rows with no label.

    Returns the clean frame and the number of scored rows dropped for a missing label.
    """
    if missing_label not in _MISSING_LABEL:
        raise ValueError(f"missing_label must be one of {_MISSING_LABEL}, got {missing_label!r}")
    missing = set(cols) - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")
    df = panel[cols].replace([np.inf, -np.inf], np.nan)
    df = df[df[score].notna() & df[date].notna()]
    # a date with no realized label at all is a holding period still open: not evidence
    realized = df.groupby(date)[label].transform(lambda s: s.notna().any())
    df = df[realized.astype(bool)]
    holes = df[label].isna()
    n_holes = int(holes.sum())
    if n_holes and missing_label == "raise":
        first = sorted(df.loc[holes, date].unique())[:3]
        raise ValueError(
            f"{n_holes} scored rows have no label on dates that are otherwise realized "
            f"(first: {first}). A name that left the universe mid-period (a delisting) "
            "still earned a return; fill it, or pass missing_label='drop' to drop these "
            "rows knowingly (the count is reported on the result).")
    return df[~holes], n_holes


# --------------------------------------------------------------------------- decay

@dataclass(frozen=True)
class BandDecay:
    """Per-band outcome of a cross-sectional ranking, best band first.

    ``table`` columns, one row per band (band 1 = highest scores):

      ranks         e.g. "1-5"
      mean_return   mean over dates of the band's equal-weight period return
      t_stat        mean_return / standard error of the per-date band returns
                    (i.i.d. across dates; overlapping labels or a sticky score
                    overstate it, see random_selection_null)
      win_rate      fraction of (date, name) observations with label > 0, pooled
      payoff        mean winner / |mean loser|, pooled over (date, name)
      periods       dates compared (the same for every band)
      names         total (date, name) observations

    Every band is measured on the same dates, the ones where all of them are full, so
    the spread between bands cannot come from bands covering different years.
    ``dates_dropped`` counts the realized dates that could not fill every band;
    ``labels_dropped`` counts scored rows dropped under ``missing_label='drop'``.

    ``monotonic`` is True when ``mean_return`` is non-increasing from band 1 down.
    (The table runs best-first, the reverse of ``DecayTable``, which runs Q1 upward.)
    """

    table: pd.DataFrame
    monotonic: bool
    band_size: int
    dates_dropped: int = 0
    labels_dropped: int = 0

    @property
    def spread(self) -> float:
        """Top band minus bottom band mean period return."""
        m = self.table["mean_return"]
        return float(m.iloc[0] - m.iloc[-1])

    @property
    def cliff_share(self) -> float:
        """Share of the top-to-bottom spread lost in the first step (band 1 to 2).

        About 1/(bands-1) is an even slope; near 1 means the whole edge sits in the
        top band and the rest of the ranking is indistinguishable. Undefined (nan)
        when the spread is not positive. Can exceed 1 when band 2 sits below the
        bottom band.
        """
        m = self.table["mean_return"].to_numpy()
        if len(m) < 2 or self.spread <= 0:
            return float("nan")
        return float((m[0] - m[1]) / self.spread)


def rank_band_decay(panel: pd.DataFrame, *, date: str = "date", score: str = "score",
                    label: str = "label", band_size: int = 5,
                    n_bands: int | None = 6,
                    missing_label: str = "raise") -> BandDecay:
    """Rank names within each date (highest ``score`` = rank 1), group ranks into
    bands of ``band_size`` and report each band as its own equal-weight portfolio.

    Only dates on which all ``n_bands`` bands are full count, so a thin early
    universe can neither fill band 6 with two names nor leave band 6 averaged over
    later years than band 1. ``n_bands=None`` uses as many bands as the thinnest
    date fills, which drops no date.

    Unlike :func:`crucible.ml.quantile_decay` (win rate per quantile of a pooled
    score), this ranks *within* each date and reports return, its t-stat, win rate
    and payoff. For a positive-skew strategy the edge can live in payoff while win
    rate stays flat.
    """
    if band_size < 1:
        raise ValueError("band_size must be >= 1")
    if n_bands is not None and n_bands < 1:
        raise ValueError("n_bands must be >= 1 or None")
    df, labels_dropped = _clean(panel, [date, score, label], date, score, label,
                                missing_label)
    if df.empty:
        raise ValueError("no valid rows in panel")

    full_bands = df.groupby(date).size() // band_size
    bands = int(full_bands.min()) if n_bands is None else n_bands
    if bands < 1:
        raise ValueError(f"some date has fewer than band_size={band_size} names; "
                         "pass n_bands to drop the thin dates instead")
    keep = full_bands.index[full_bands >= bands]
    if len(keep) == 0:
        raise ValueError(f"no date has {bands} full bands of {band_size} names")
    dates_dropped = int(len(full_bands) - len(keep))

    df = df[df[date].isin(keep)]
    df = df.assign(_rank=df.groupby(date)[score].rank(method="first", ascending=False))
    df["_band"] = ((df["_rank"] - 1) // band_size + 1).astype(int)
    df = df[df["_band"] <= bands]

    per_period = df.groupby(["_band", date])[label].mean()
    rows = []
    for b, g in df.groupby("_band"):
        lab = g[label].to_numpy()
        pp = per_period.loc[b]
        wins, losses = lab[lab > 0], lab[lab <= 0]
        payoff = (wins.mean() / abs(losses.mean())
                  if len(wins) and len(losses) and losses.mean() != 0 else np.nan)
        rows.append({
            "band": int(b),
            "ranks": f"{(b - 1) * band_size + 1}-{b * band_size}",
            "mean_return": float(pp.mean()),
            "t_stat": float(pp.mean() / (pp.std(ddof=1) / np.sqrt(len(pp))))
                      if len(pp) > 1 and pp.std(ddof=1) > 0 else float("nan"),
            "win_rate": float((lab > 0).mean()),
            "payoff": float(payoff),
            "periods": int(pp.size),
            "names": int(len(lab)),
        })
    table = pd.DataFrame(rows).sort_values("band").reset_index(drop=True)
    monotonic = bool(np.all(np.diff(table["mean_return"].to_numpy()) <= 0))
    return BandDecay(table=table, monotonic=monotonic, band_size=band_size,
                     dates_dropped=dates_dropped, labels_dropped=labels_dropped)


# ---------------------------------------------------------------------------- null

@dataclass(frozen=True)
class SelectionNull:
    """Top-N against the same rule run on name-permuted scores.

    ``observed``   mean over dates of the top-N equal-weight period return
    ``null_mean``  mean of the permuted runs' averages
    ``p_value``    P(permuted average >= observed), one-sided, with +1 smoothing,
                   uncorrected for how many ``top_n`` values were tried
    ``percentile`` where the observed sits in the null distribution (0-100)
    ``excess``     per-date series: top-N return minus that date's universe mean,
                   ordered by date. Descriptive; a time-block bootstrap of it does
                   NOT repair a sticky score picking the same names every date
                   (the dependence is across names, not only across time)
    ``labels_dropped`` scored rows dropped under ``missing_label='drop'``
    """

    observed: float
    null_mean: float
    p_value: float
    percentile: float
    top_n: int
    n_sims: int
    periods: int
    excess: pd.Series
    labels_dropped: int = 0

    def __str__(self) -> str:
        return (f"top-{self.top_n} vs name-permuted top-{self.top_n} over {self.periods} "
                f"periods: observed {self.observed:+.4f}, null {self.null_mean:+.4f}, "
                f"p = {self.p_value:.4f} ({self.percentile:.1f}th pct, "
                f"{self.n_sims} sims)")


def random_selection_null(panel: pd.DataFrame, *, top_n: int, date: str = "date",
                          name: str = "name", score: str = "score", label: str = "label",
                          n_sims: int = 2000, seed: int = 0,
                          missing_label: str = "raise") -> SelectionNull:
    """Does picking the top ``top_n`` by ``score`` beat the same rule when the scores
    carry no information about the names they are attached to?

    Each simulation draws one random permutation of the names and gives every name
    another name's whole score path, then picks the top ``top_n`` on every date
    exactly as the observed rule does. Each name keeps its own return path, so the
    null keeps everything the observed selection has except the link between score
    and name: market-wide moves, a persistent score picking the same names month
    after month, overlapping holding periods, and names that simply drifted up for
    reasons the score does not know about. Drawing names independently on each date
    instead keeps none of the last three, and on a sticky score it passes a
    worthless ranking several times more often than its nominal rate.

    On a date where a name's permuted partner has no score, that name is filled in at
    random below the permuted picks, so every simulation is scored on the same dates
    as the observation. On a balanced panel no fill happens and the test is an exact
    permutation test (names exchangeable under the null).

    Dates with fewer than ``top_n`` eligible names are dropped. Score ties break by
    name order. The question is whether the *ordering* adds value over membership of
    the universe: a universe that simply went up passes "beats zero" but not this.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    df, labels_dropped = _clean(panel, [date, name, score, label], date, score, label,
                                missing_label)
    if df.duplicated([date, name]).any():
        raise ValueError(f"panel has duplicate ({date}, {name}) rows")
    counts = df.groupby(date).size()
    df = df[df[date].isin(counts.index[counts >= top_n])]
    if df.empty:
        raise ValueError(f"no date has at least top_n={top_n} eligible names")

    S = df.pivot(index=date, columns=name, values=score).sort_index()
    L = df.pivot(index=date, columns=name, values=label).reindex_like(S)
    dates = S.index
    s, lab = S.to_numpy(float), L.to_numpy(float)
    present = ~np.isnan(s)
    n_dates, n_names = s.shape
    rows = np.arange(n_dates)[:, None]

    # within-date percentile of each score, ties by name order; ordering among any
    # subset of names is preserved, so permuting these is permuting the scores
    pct = S.rank(axis=1, method="first", pct=True).to_numpy(float)
    lab0 = np.where(present, lab, 0.0)

    def top_mean(key: np.ndarray) -> np.ndarray:
        idx = np.argpartition(-key, top_n - 1, axis=1)[:, :top_n]
        return lab0[rows, idx].mean(axis=1)

    obs = top_mean(np.where(present, pct, -np.inf))
    uni = np.nanmean(np.where(present, lab, np.nan), axis=1)
    observed = float(obs.mean())

    rng = np.random.default_rng(seed)
    sims = np.empty(n_sims)
    for k in range(n_sims):
        perm = rng.permutation(n_names)
        key = 1.0 + pct[:, perm]                 # permuted picks sit in (1, 2]
        fill = present & np.isnan(key)
        if fill.any():                           # random fill sits in [0, 1)
            key[fill] = rng.random(int(fill.sum()))
        key[~present] = -np.inf
        sims[k] = top_mean(key).mean()

    p_value = float((np.sum(sims >= observed) + 1) / (n_sims + 1))
    excess = pd.Series(obs - uni, index=pd.Index(dates, name=date), name="excess")
    return SelectionNull(observed=observed, null_mean=float(sims.mean()), p_value=p_value,
                         percentile=float((sims < observed).mean() * 100),
                         top_n=top_n, n_sims=n_sims, periods=n_dates, excess=excess,
                         labels_dropped=labels_dropped)
