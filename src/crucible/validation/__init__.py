"""
crucible.validation — does the edge survive out of sample?

The honest tests you run once you have a TradeLog (or a strategy + prices):

  holdout        an early-train / late-confirm temporal split, leakage-controlled
  walk_forward   Pardo anchored/rolling walk-forward with per-fold efficiency
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
from crucible.validation.holdout import holdout, split_train_test, HoldoutResult
from crucible.validation.walk_forward import (
    walk_forward, WalkForwardResult, Fold,
)
from crucible.validation.permutation import (
    sign_permutation_pvalue, sidak_correction, whites_reality_check, spa_test,
)
from crucible.validation.pbo import (
    pbo_cscv, PBOResult, deflated_sharpe, DeflatedSharpe,
)
from crucible.validation.search_space import SearchSpaceLog
from crucible.validation.gate import Gate, GateCheck, Gauntlet
from crucible.validation.diagnostics import fold_dispersion, walk_forward_efficiency
from crucible.validation.thresholds import Thresholds
from crucible.validation.gauntlet import (
    run_gauntlet, gate_real, gate_strong, gate_durable, gate_general,
)

__all__ = [
    "holdout", "split_train_test", "HoldoutResult",
    "walk_forward", "WalkForwardResult", "Fold",
    "sign_permutation_pvalue", "sidak_correction", "whites_reality_check", "spa_test",
    "pbo_cscv", "PBOResult", "deflated_sharpe", "DeflatedSharpe",
    "SearchSpaceLog",
    "Gate", "GateCheck", "Gauntlet",
    "fold_dispersion", "walk_forward_efficiency",
    "Thresholds",
    "run_gauntlet", "gate_real", "gate_strong", "gate_durable", "gate_general",
]
