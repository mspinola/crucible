"""crucible.validation.gauntlet — the capital-free edge-validation gauntlet.

A trade log runs a gauntlet of named gates, each proving one property. crucible
owns the capital-free middle of the sequence; the ends are yours to bring or to
hand off:

    DECLARE   preamble  — a mechanical rule + a log of every variant you tried
    CLEAN     preamble  — leakage-controlled construction (use holdout/walk_forward)
    ──────────────── the gauntlet crucible computes ────────────────
    REAL      — distinguishable from noise, corrected for the search
    STRONG    — economically meaningful at the CI lower bound
    DURABLE   — holds out-of-sample over time (walk-forward)
    GENERAL   — travels to markets it wasn't built on (optional)
    ────────────────────────────────────────────────────────────────
    SURVIVE   handoff   — capital survivability (out of scope — hand the
                          surviving TradeLog to a capital-aware tool)

Every gate is an audited :class:`Gate` (hard checks AND to the verdict; soft
checks inform but never gate). Nothing here touches capital or an equity curve.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from crucible.edge import (
    TradeLog, edge_report, expectancy, sqn,
    bootstrap_metric_cis, random_entry_null,
)
from crucible.validation.permutation import (
    sign_permutation_pvalue, sidak_correction, whites_reality_check,
)
from crucible.validation.diagnostics import fold_dispersion, walk_forward_efficiency
from crucible.validation.gate import Gate, Gauntlet
from crucible.validation.thresholds import Thresholds


def _infer_hold(trades: TradeLog, default: int = 20) -> int:
    """Mean holding period from the log's bars_held, if present — so the random
    null holds trades for the same span the strategy did."""
    b = trades.col("bars_held")
    if b is not None and len(b):
        vals = np.asarray(b, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            return max(1, int(round(float(vals.mean()))))
    return default


def gate_real(trades: TradeLog, *, prices: Optional[pd.DataFrame] = None,
              side: str = "long", hold: Optional[int] = None,
              tp: float = 2.0, sl: float = 1.0,
              variant_returns: Optional[Dict[str, object]] = None,
              n_variants: Optional[int] = None,
              thr: Thresholds = Thresholds()) -> Gate:
    """REAL — is the edge distinguishable from noise, corrected for the search?

    Significance is the hard check. If you pass `variant_returns` (every variant
    you searched, discards included), it uses White's Reality Check across them;
    else a sign-permutation p-value, Šidák-corrected when you declare
    `n_variants`. When `prices` are given it also requires the expectancy to beat
    the 95th percentile of random-entry books on the *same* prices (crucible's
    R-consistent "beat coin-flip timing" null); without prices that check is
    recorded as a soft skip.
    """
    g = Gate("REAL")

    if variant_returns:
        rc = whites_reality_check(variant_returns, n_permutations=thr.n_perm, seed=thr.seed)
        p = rc["corrected_pvalue"]
        g.add("reality_check_pvalue", p < thr.alpha, value=p, threshold=thr.alpha,
              detail=f"White's Reality Check across {rc['n_variants']} searched variants "
                     f"(best: {rc['best_variant']})")
    else:
        p_raw = sign_permutation_pvalue(trades, n_permutations=thr.n_perm, seed=thr.seed)
        if n_variants and n_variants > 1:
            p = sidak_correction(p_raw, n_variants)
            g.add("corrected_pvalue", p < thr.alpha, value=p, threshold=thr.alpha,
                  detail=f"sign-permutation p={p_raw:.4f}, Šidák-corrected for {n_variants} variants")
        else:
            p = p_raw
            g.add("permutation_pvalue", p < thr.alpha, value=p, threshold=thr.alpha,
                  detail="sign-permutation, uncorrected — declare a variant count for the "
                         "search correction if you searched more than one")

    if prices is not None and trades.n:
        h = hold if hold is not None else _infer_hold(trades)
        null = random_entry_null(prices, side=side, n_entries=trades.n, hold=h,
                                 tp=tp, sl=sl, n_sims=thr.n_random_sims, seed=thr.seed)
        finite = null[np.isfinite(null)]
        if len(finite):
            obs = expectancy(trades.r)
            bar = float(np.percentile(finite, 95))
            beaten = float((finite < obs).mean()) * 100
            g.add("beats_random_timing", obs > bar, value=obs, threshold=bar,
                  detail=f"expectancy beats {beaten:.0f}% of random-entry books on the "
                         f"same prices (needs to beat the 95th percentile)")
        else:
            g.add("beats_random_timing", False, hard=False,
                  detail="skipped — random-entry null produced no trades")
    else:
        g.add("beats_random_timing", False, hard=False,
              detail="skipped — no price series provided")

    return g


def gate_strong(trades: TradeLog, *, thr: Thresholds = Thresholds()) -> Gate:
    """STRONG — economically meaningful at the CI lower bound?

    Gates expectancy and profit factor on their bootstrap CI *lower* bound (not
    the point estimate — that's the fix for "PF 1.37 on 60 trades looked clean").
    SQN's lower bound is a soft/aspirational bar, and the structural excursion
    metrics (when the log carries mfe/mae/bars_held) are soft, informational.
    """
    g = Gate("STRONG")
    cis = bootstrap_metric_cis(trades, n_boot=thr.n_boot, alpha=thr.alpha, seed=thr.seed)
    exp, pf, s = cis["expectancy"], cis["profit_factor"], cis["sqn"]

    g.add("expectancy_ci_lower", exp.low > thr.min_expectancy_ci_low,
          value=exp.low, threshold=thr.min_expectancy_ci_low, detail=f"point={exp.point:+.3f}")
    g.add("profit_factor_ci_lower", pf.low > thr.min_profit_factor_ci_low,
          value=pf.low, threshold=thr.min_profit_factor_ci_low, detail=f"point={pf.point:.2f}")
    g.add("sqn_ci_lower", s.low > thr.min_sqn_ci_low, hard=False,
          value=s.low, threshold=thr.min_sqn_ci_low, detail=f"point={s.point:.2f}")

    er = edge_report(trades)
    if er.excursion_ratio is not None:
        g.add("excursion_ratio", er.excursion_ratio >= thr.min_excursion_ratio, hard=False,
              value=er.excursion_ratio, threshold=thr.min_excursion_ratio)
    if er.time_asymmetry is not None:
        g.add("time_asymmetry", er.time_asymmetry >= thr.min_time_asymmetry, hard=False,
              value=er.time_asymmetry, threshold=thr.min_time_asymmetry)
    if er.exit_efficiency is not None:
        g.add("exit_efficiency", er.exit_efficiency >= thr.min_exit_efficiency, hard=False,
              value=er.exit_efficiency, threshold=thr.min_exit_efficiency)

    return g


def gate_durable(wf, *, thr: Thresholds = Thresholds()) -> Gate:
    """DURABLE — does the edge hold out-of-sample over time?

    Reads a `WalkForwardResult`: the aggregate return-based WFE must sit in its
    accept band (both extremes reject), and the folds must not be chaotic — a
    majority individually tradable (SQN>0) with bounded dispersion.
    """
    g = Gate("DURABLE")

    wfe = walk_forward_efficiency(
        [f.wfe for f in wf.folds],
        reject_low=thr.wfe_reject_low, reject_high=thr.wfe_reject_high,
        target_low=thr.wfe_target_low, target_high=thr.wfe_target_high)
    if wfe is not None:
        g.add("wfe_aggregate", wfe["passes"], value=wfe["aggregate_wfe"],
              threshold=(wfe["reject_low"], wfe["reject_high"]),
              detail=f"mean annualized OOS/IS return over {wfe['n_folds']} folds; "
                     f"reject <{wfe['reject_low']:.0%} or >{wfe['reject_high']:.0%}")
        g.add("wfe_in_target_band", wfe["in_target_band"], hard=False,
              value=wfe["aggregate_wfe"],
              threshold=(wfe["target_low"], wfe["target_high"]),
              detail=f"healthy band {wfe['target_low']:.0%}–{wfe['target_high']:.0%}")
    else:
        g.add("wfe_aggregate", False, detail="no folds to evaluate")

    disp = fold_dispersion([sqn(f.oos_trades.r) for f in wf.folds],
                           min_tradable_pct=thr.min_folds_tradable_pct,
                           max_cv=thr.max_fold_sqn_cv)
    if disp is not None:
        g.add("fold_dispersion", disp["passes"], value=disp["fold_sqn_cv"],
              threshold=disp["max_cv"],
              detail=f"{disp['pct_folds_tradable']*100:.0f}% of {disp['n_folds']} folds "
                     f"tradable (SQN>0), mean fold SQN={disp['mean_fold_sqn']:.2f}")
    else:
        g.add("fold_dispersion", False, detail="no fold SQN data")

    return g


def gate_general(trade_logs: Dict[str, TradeLog], *,
                 thr: Thresholds = Thresholds()) -> Gate:
    """GENERAL — does the edge travel to markets it wasn't built on?

    Testing many markets is itself a search, so the pass bar is the cross-market
    Reality Check: treat each market's per-trade returns as a variant and require
    the best market to beat the distribution of the best under no skill, across
    every market tested. Pass every market attempted (including the failures) —
    which are development vs. held-out is your record to keep, not crucible's.
    """
    g = Gate("GENERAL")
    variant_returns = {k: tl.r for k, tl in trade_logs.items()}
    rc = whites_reality_check(variant_returns, n_permutations=thr.n_perm, seed=thr.seed)
    g.add("cross_market_reality_check", rc["corrected_pvalue"] < thr.alpha,
          value=rc["corrected_pvalue"], threshold=thr.alpha,
          detail=f"best market '{rc['best_variant']}' across {rc['n_variants']} tested")
    return g


def run_gauntlet(trades: TradeLog, *, prices: Optional[pd.DataFrame] = None,
                 wf=None, trade_logs: Optional[Dict[str, TradeLog]] = None,
                 side: str = "long", hold: Optional[int] = None,
                 tp: float = 2.0, sl: float = 1.0,
                 variant_returns: Optional[Dict[str, object]] = None,
                 n_variants: Optional[int] = None,
                 thr: Thresholds = Thresholds()) -> Gauntlet:
    """Run the gauntlet on a trade log. REAL and STRONG always run; DURABLE runs
    when a `WalkForwardResult` is supplied, GENERAL when a {market: TradeLog} map
    is. The gauntlet passes only if every gate it ran passes."""
    gauntlet = Gauntlet()
    gauntlet.add(gate_real(trades, prices=prices, side=side, hold=hold, tp=tp, sl=sl,
                           variant_returns=variant_returns, n_variants=n_variants, thr=thr))
    gauntlet.add(gate_strong(trades, thr=thr))
    if wf is not None:
        gauntlet.add(gate_durable(wf, thr=thr))
    if trade_logs is not None:
        gauntlet.add(gate_general(trade_logs, thr=thr))
    return gauntlet
