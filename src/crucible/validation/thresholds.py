"""crucible.validation.thresholds — the bars the gauntlet gates against.

One frozen dataclass of defaults so every gate reads from a single, overridable
place. The defaults are deliberately conservative — tuned to fail a marginal
result rather than pass it, because a false PASS costs capital while a false FAIL
only costs a redesign. Pass a customized `Thresholds` to any gate to retune.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # ── REAL — search-corrected significance ────────────────────────────────
    alpha: float = 0.05                     # corrected p-value (and CI level) bar

    # ── STRONG — edge metrics, gated on the bootstrap CI LOWER bound ─────────
    min_expectancy_ci_low: float = 0.0      # hard: expectancy lower bound > 0
    min_profit_factor_ci_low: float = 1.25  # hard: PF lower bound > 1.25
    min_sqn_ci_low: float = 1.6             # soft: aspirational (Van Tharp "tradable")
    # structural excursion family — soft, informational (needs mfe/mae/bars_held)
    min_excursion_ratio: float = 1.2
    min_time_asymmetry: float = 2.0
    min_exit_efficiency: float = 0.5

    # ── DURABLE — walk-forward robustness ───────────────────────────────────
    wfe_reject_low: float = 0.30            # WFE below this = fragile (reject)
    wfe_reject_high: float = 1.00           # WFE above this = too-good-to-be-true (reject)
    wfe_target_low: float = 0.50            # healthy band (soft)
    wfe_target_high: float = 0.80
    min_folds_tradable_pct: float = 0.5     # majority of folds must be tradable (SQN>0)
    max_fold_sqn_cv: float = 2.0            # cap on fold-SQN coefficient of variation

    # ── resampling budgets / determinism ────────────────────────────────────
    n_boot: int = 10_000                    # bootstrap draws
    n_perm: int = 5_000                     # permutation / reality-check draws
    n_random_sims: int = 1_000              # random-entry null simulations
    seed: int = 0
