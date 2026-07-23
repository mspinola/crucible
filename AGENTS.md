# AGENTS.md

Guidance for AI coding agents working on **or with** crucible. Written by a human, kept short
on purpose. If you are an agent, treat this as constraints, not suggestions, and propose edits
to a maintainer rather than expanding it yourself.

---

## If you are USING crucible with an agent

crucible exists to be the one part of your research loop that an AI cannot talk out of a verdict.
It reasons over a trade log with statistics, never over an equity curve, so it cannot be tempted
to call a big number an edge. That only holds if you keep the roles apart.

- **Do not let the agent judge its own work.** An agent may draft a hypothesis, write the signal,
  and build the trade log. It must not then declare the result good. Hand the `TradeLog` to the
  gate and read back what the gate says, including the checks that fail. The moment the author
  becomes the judge, the verdict is worthless.

- **Log every variant the agent tries, including the discards.** Agents make good-looking
  strategies free to produce, which is exactly what breaks a naive significance test. Use
  `crucible.validation.SearchSpaceLog(scope=..., path=...)`, call `record(...)` for every
  configuration tried and `mark_selected(...)` for the winner, and pass the log to
  `run_gauntlet(..., n_variants=log)`. The correction is only honest if its denominator counts
  the whole search. A smaller reported count is a prettier p-value and a broken gate.

- **Build the log leakage-free by construction.** Prefer `crucible.validation.holdout` or
  `walk_forward`, which enforce purge and embargo for you. If the agent imports a log from a
  broker or another backtester, leakage-freedom is unproven, and crucible cannot see it. Say so.

- **Ask the agent to attack a passing edge, not just find one.** A result that survives only at
  its exact parameters is fragile. Perturb a passed config and rerun the gate. This is diagnosis,
  run after a pass to stress it, not a second search for a config that happens to clear the bar.

The alpha is not in the prompt. It is in the process you build around it. crucible is one honest
referee in that process, not the whole of it.

## If you are CONTRIBUTING to crucible

- **The public function names are the contract.** The gate names (`edge_report`, `reality_check`,
  `run_gauntlet`, `holdout`, `walk_forward`, `pbo`, `bootstrap_ci`, `SearchSpaceLog`, and the
  rest of the documented surface) are what users pin against, and the package declares them in
  `__all__`. Do not rename, move, or change the meaning of a public symbol, or drop one from
  `__all__`, without a deprecation path.

- **A hard check is load-bearing.** The gauntlet's value is that a failing hard check cannot be
  waived by someone who likes the strategy. Do not add a flag that softens a hard check into a
  warning, and do not turn a soft/informational metric into something that can gate. If a
  threshold needs to change, change it in the open with a rationale, not behind an option.

- **Determinism is a feature.** Same input, same seed, same verdict. Do not introduce
  nondeterminism into a gate path. Randomized procedures (bootstrap, permutation) take an explicit
  seed and must reproduce.

- **Keep crucible capital-free.** crucible never sees currency, position sizing, or an equity
  curve. Those live downstream (see the capital-aware layers). A change that pulls account-level
  concepts into crucible breaks the separation the whole design depends on.

- **Docs and tests move with the code.** Public behavior changes update the tutorial and the
  relevant docs in `docs/` in the same change. Every claim in the tutorial is expected to run.
