"""Permutation / data-mining tests — is the edge real, and did you find it by
searching too many variants?

  sign_permutation_pvalue   Masters' sign-flip test on one return series
                            (public-domain stand-in for White's Reality Check)
  sidak_correction          conservative multiple-testing correction from a count
  whites_reality_check      max-statistic permutation across EVERY variant tried,
                            already corrected for the size of the search space

All operate on plain return arrays or TradeLogs — no capital, no COT.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np

from crucible.edge.trade_log import TradeLog

Returns = Union[Sequence[float], TradeLog]


def _as_returns(r: Returns) -> np.ndarray:
    arr = r.r if isinstance(r, TradeLog) else np.asarray(r, dtype=float)
    return arr[~np.isnan(arr)]


def sign_permutation_pvalue(returns: Returns, n_permutations: int = 5000,
                            seed: Optional[int] = 0) -> float:
    """Monte-Carlo sign-permutation test (Tim Masters).

    Null: no directional skill, so each trade's sign is as likely flipped as not.
    Flip signs at random and ask how often the permuted mean matches/beats the
    observed mean. Returns the UNCORRECTED p-value — pass it through
    `sidak_correction` with your variant count, or use `whites_reality_check`
    when you have every variant's return series."""
    r = _as_returns(returns)
    if len(r) == 0:
        return 1.0
    observed = r.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(r)))
    permuted = (signs * np.abs(r)).mean(axis=1)
    return float((np.sum(permuted >= observed) + 1) / (n_permutations + 1))


def sidak_correction(p_raw: float, n_variants: int) -> float:
    """Šidák: probability the best of `n_variants` independent searches would
    produce a p-value this small by chance. corrected = 1 - (1 - p_raw)^n.

    The conservative fallback when you only know the COUNT of variants tried;
    with full return series available, `whites_reality_check` is less punishing."""
    if n_variants < 1:
        raise ValueError(f"n_variants must be >= 1, got {n_variants}")
    p_raw = min(max(p_raw, 0.0), 1.0)
    return float(1.0 - (1.0 - p_raw) ** n_variants)


def whites_reality_check(variant_returns: Dict[str, Returns],
                         n_permutations: int = 5000,
                         seed: Optional[int] = 0) -> dict:
    """White's Reality Check via sign permutation across the whole search space.

    Each permutation flips signs for every variant and records the BEST mean
    under the null; the observed best variant is compared against that
    distribution of best performers. Taking the max inside each permutation is
    what corrects for how many variants you tried.

    Pass EVERY variant you searched, including the discards — omitting the losers
    biases the test toward a false pass. Returns best_variant, observed_best_mean,
    corrected_pvalue, n_variants."""
    cleaned = {k: _as_returns(v) for k, v in variant_returns.items()}
    cleaned = {k: v for k, v in cleaned.items() if len(v)}
    if not cleaned:
        return {"best_variant": None, "observed_best_mean": float("nan"),
                "corrected_pvalue": 1.0, "n_variants": 0}

    observed = {k: v.mean() for k, v in cleaned.items()}
    best_variant = max(observed, key=observed.get)
    observed_best = observed[best_variant]

    rng = np.random.default_rng(seed)
    null_best = np.empty(n_permutations)
    for i in range(n_permutations):
        best = -np.inf
        for v in cleaned.values():
            signs = rng.choice([-1.0, 1.0], size=len(v))
            best = max(best, float((signs * np.abs(v)).mean()))
        null_best[i] = best

    p = float((np.sum(null_best >= observed_best) + 1) / (n_permutations + 1))
    return {"best_variant": best_variant, "observed_best_mean": float(observed_best),
            "corrected_pvalue": p, "n_variants": len(cleaned)}


def spa_test(variant_returns: Dict[str, Returns],
             n_permutations: int = 5000,
             seed: Optional[int] = 0) -> dict:
    """Hansen's Superior Predictive Ability test — White's Reality Check's more
    powerful successor (Hansen 2005). Same max-statistic-under-the-null idea, with
    the two upgrades that matter when the variants are heterogeneous:

      * STUDENTIZED — each variant's mean is divided by its standard error before
        the max, so a noisy / low-N variant can't dominate the statistic just by
        being volatile (`whites_reality_check` compares raw means).
      * INFERIOR VARIANTS EXCLUDED (the consistent SPA_c recentering) — variants
        whose studentized mean is clearly below zero (< -sqrt(2·ln ln N)) are dropped
        from the null max, so padding the pool with junk no longer weakens the test —
        WRC's known flaw. The well-known consequence: **SPA p <= WRC p** (more power).

    This is `spa_test` ADDED alongside `whites_reality_check`, not a replacement:
    WRC is the more conservative number, SPA the more powerful one — report both and
    adopt the extra power deliberately. Same sign-permutation resampling/conventions.
    Returns best_variant, observed_max_t (studentized), corrected_pvalue, n_variants
    (kept), n_excluded (dropped as inferior)."""
    cleaned = {k: _as_returns(v) for k, v in variant_returns.items()}
    cleaned = {k: v for k, v in cleaned.items() if len(v) >= 2}
    _none = {"best_variant": None, "observed_max_t": float("nan"),
             "corrected_pvalue": 1.0, "n_variants": 0, "n_excluded": 0}
    if not cleaned:
        return _none

    # studentize each variant (t = mean / SE) and drop the clearly-inferior ones.
    kept = {}
    for k, v in cleaned.items():
        n = len(v)
        se = float(v.std(ddof=1)) / np.sqrt(n)
        if se <= 0:
            continue
        t = float(v.mean()) / se
        if t >= -np.sqrt(2.0 * np.log(np.log(max(n, 3)))):   # SPA_c exclusion band
            kept[k] = (v, se, t)
    if not kept:
        return {**_none, "n_excluded": len(cleaned)}

    best_variant = max(kept, key=lambda k: kept[k][2])
    observed_max_t = kept[best_variant][2]

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_permutations)
    for i in range(n_permutations):
        m = -np.inf
        for v, se, _ in kept.values():
            signs = rng.choice([-1.0, 1.0], size=len(v))
            m = max(m, float((signs * np.abs(v)).mean()) / se)
        null_max[i] = m

    p = float((np.sum(null_max >= observed_max_t) + 1) / (n_permutations + 1))
    return {"best_variant": best_variant, "observed_max_t": float(observed_max_t),
            "corrected_pvalue": p, "n_variants": len(kept),
            "n_excluded": len(cleaned) - len(kept)}
