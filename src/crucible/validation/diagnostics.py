"""crucible.validation.diagnostics — fold-level robustness diagnostics.

Capital-free reads on a walk-forward run that a single aggregate number hides:

  fold_dispersion            are the individual folds tradable and consistent, or
                             does a smooth average paper over chaotic folds?
  walk_forward_efficiency    the aggregate return-based WFE across folds, with
                             BOTH extremes flagged (too-low = fragile, too-high =
                             too-good-to-be-true).

Both take plain per-fold sequences (SQNs, WFEs) — no capital, no equity curve.
They feed the DURABLE gate; on their own they're just diagnostics.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def fold_dispersion(fold_sqns: Sequence[float], min_tradable_pct: float = 0.5,
                    max_cv: float = 2.0) -> Optional[dict]:
    """Fold-dispersion check: does the walk-forward hold up fold by fold?

    An aggregate efficiency can average chaotic folds into a plausible-looking
    number. This looks at the folds individually:

      - what fraction are themselves tradable (SQN > 0)?
      - how dispersed is fold-level SQN (coefficient of variation)?

    High dispersion is a rejection signal in its own right, independent of the
    average. Returns None on empty input.

    max_cv bounds the |coefficient of variation| of fold SQN. CV is reported as
    inf (a fail) when mean fold SQN is ~0 — that means the folds carry no
    consistent edge, which is exactly the failure the check exists to catch.
    """
    sqns = np.asarray([s for s in fold_sqns if s is not None], dtype=float)
    sqns = sqns[~np.isnan(sqns)]
    if len(sqns) == 0:
        return None

    pct_tradable = float((sqns > 0).mean())
    mean_sqn = sqns.mean()
    std_sqn = sqns.std(ddof=1) if len(sqns) > 1 else 0.0
    cv = float("inf") if abs(mean_sqn) < 1e-12 else float(abs(std_sqn / mean_sqn))

    return {
        "n_folds": len(sqns),
        "pct_folds_tradable": pct_tradable,
        "mean_fold_sqn": float(mean_sqn),
        "fold_sqn_cv": cv,
        "min_tradable_pct": min_tradable_pct,
        "max_cv": max_cv,
        "passes": bool(pct_tradable >= min_tradable_pct and cv <= max_cv),
    }


def walk_forward_efficiency(fold_wfes: Sequence[float],
                            reject_low: float = 0.30, reject_high: float = 1.00,
                            target_low: float = 0.50, target_high: float = 0.80
                            ) -> Optional[dict]:
    """Aggregate return-based Walk-Forward Efficiency across folds (Pardo).

    Per-fold WFE = annualized OOS return / annualized IS return (one value per
    fold; crucible's `walk_forward` reports these). The aggregate is the mean.

    Both extremes are hard rejects, not just a low value:
      - WFE < reject_low: the edge barely survives moving IS -> OOS.
      - WFE > reject_high: OOS wildly outrunning IS — the "too good to be true"
        case, almost always curve-fit luck or a near-zero IS baseline inflating
        the ratio, not genuine robustness.
    The [target_low, target_high] band (50-80%) is the healthy target and is
    reported separately as a soft signal. Returns None on empty input.
    """
    vals = np.asarray([w for w in fold_wfes if w is not None], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None

    agg = float(vals.mean())
    return {
        "n_folds": len(vals),
        "aggregate_wfe": agg,
        "reject_low": reject_low,
        "reject_high": reject_high,
        "target_low": target_low,
        "target_high": target_high,
        "in_target_band": bool(target_low <= agg <= target_high),
        "passes": bool(reject_low <= agg <= reject_high),
    }
