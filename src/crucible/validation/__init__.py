"""
crucible.validation — does the edge survive out of sample?

The honest tests you run once you have a TradeLog (or a strategy + prices):

  holdout        an early-train / late-confirm temporal split, leakage-controlled
  walk_forward   Pardo anchored/rolling walk-forward with per-fold efficiency
  permutation    sign-permutation p-value, data-mining correction, White's
                 Reality Check across every variant you searched
  gate           an audited, un-overridable pass/fail gate (and a Gauntlet of them)
  diagnostics    fold-level robustness reads (dispersion, walk-forward efficiency)
"""
from crucible.validation.holdout import holdout, split_train_test, HoldoutResult
from crucible.validation.walk_forward import (
    walk_forward, WalkForwardResult, Fold,
)
from crucible.validation.permutation import (
    sign_permutation_pvalue, sidak_correction, whites_reality_check,
)
from crucible.validation.gate import Gate, GateCheck, Gauntlet
from crucible.validation.diagnostics import fold_dispersion, walk_forward_efficiency

__all__ = [
    "holdout", "split_train_test", "HoldoutResult",
    "walk_forward", "WalkForwardResult", "Fold",
    "sign_permutation_pvalue", "sidak_correction", "whites_reality_check",
    "Gate", "GateCheck", "Gauntlet",
    "fold_dispersion", "walk_forward_efficiency",
]
