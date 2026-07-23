"""
crucible.validation — does the edge survive out of sample?

The honest tests you run once you have a TradeLog (or a strategy + prices):

  holdout        an early-train / late-confirm temporal split, leakage-controlled
                 (segmented_holdout runs the same split sliced by a group column;
                 full_sample is the whole-book, in-sample counterpart — no split)
  walk_forward   Pardo anchored/rolling walk-forward with per-fold efficiency
  windows        windowed_segments — a descriptive (segment × era) metric grid
                 over an existing log (no refit); shows where/when the edge lived
  permutation    sign-permutation p-value, data-mining correction, White's
                 Reality Check + Hansen's SPA (its more powerful successor)
                 across every variant you searched
  pbo            probability of backtest overfitting (CSCV) + deflated Sharpe —
                 how much the ACT OF SELECTING the best config overfit
  search_space   the search ledger — an honest N for the data-mining correction,
                 counting every variant tried (not just the winner you kept)
  gate           an audited, un-overridable pass/fail gate (and a Gauntlet of them)
  diagnostics    fold-level robustness reads (dispersion, walk-forward efficiency)
  gauntlet       the capital-free edge-validation gauntlet: REAL / STRONG /
                 DURABLE / GENERAL gates + run_gauntlet, gated by Thresholds
"""
from crucible.validation.diagnostics import fold_dispersion, walk_forward_efficiency
from crucible.validation.gate import Gate, GateCheck, Gauntlet
from crucible.validation.gauntlet import (
    gate_durable,
    gate_general,
    gate_real,
    gate_strong,
    run_gauntlet,
)
from crucible.validation.holdout import (
    HoldoutResult,
    SegmentedHoldout,
    full_sample,
    holdout,
    segmented_holdout,
    split_train_test,
)
from crucible.validation.pbo import (
    DeflatedSharpe,
    PBOResult,
    deflated_sharpe,
    pbo_cscv,
)
from crucible.validation.permutation import (
    sidak_correction,
    sign_permutation_pvalue,
    spa_test,
    variant_count,
    whites_reality_check,
)
from crucible.validation.search_space import SearchSpaceLog
from crucible.validation.thresholds import Thresholds
from crucible.validation.walk_forward import (
    Fold,
    WalkForwardResult,
    walk_forward,
)
from crucible.validation.windows import (
    WindowCell,
    WindowedSegments,
    windowed_segments,
)

__all__ = [
    "holdout", "split_train_test", "HoldoutResult",
    "segmented_holdout", "SegmentedHoldout", "full_sample",
    "walk_forward", "WalkForwardResult", "Fold",
    "windowed_segments", "WindowedSegments", "WindowCell",
    "sign_permutation_pvalue", "sidak_correction", "variant_count",
    "whites_reality_check", "spa_test",
    "pbo_cscv", "PBOResult", "deflated_sharpe", "DeflatedSharpe",
    "SearchSpaceLog",
    "Gate", "GateCheck", "Gauntlet",
    "fold_dispersion", "walk_forward_efficiency",
    "Thresholds",
    "run_gauntlet", "gate_real", "gate_strong", "gate_durable", "gate_general",
]
