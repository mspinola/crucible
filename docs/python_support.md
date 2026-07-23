# Python support policy

This is the fleet-wide policy for `requires-python` floors across the COT/crucible
repositories. It exists so the floors stay coherent instead of drifting per repo.

## The two tiers

| Tier | Floor | Repos |
|------|-------|-------|
| Shared libraries & tooling | `>=3.10` | `cotdata`, `cotmetrics`, `crucible`, `cot-analyzer` |
| Strategy & deployment | `>=3.11` | `crucible-stack`, `npf`, `livebook` |

**Libraries and tooling** stay one minor version behind the strategy tier so they
remain broadly embeddable — they are consumed as dependencies, and a lower floor keeps
them installable in more environments.

**Strategy and deployment repos** are run as applications rather than embedded as
dependencies, so they track a newer Python and can use its features freely.

## Why not 3.9

The scientific stack moved on: recent NumPy and pandas releases require `>=3.10`, so a
3.9 floor forces old pinned versions or resolution failures. `cotdata` moved its floor
to `>=3.10` first; because `cotmetrics` and `cot-analyzer` install `cotdata` editable in
CI, their 3.9 jobs would fail at install — so they moved in lockstep. `crucible` followed
to keep the whole library tier uniform.

## When adding or bumping a repo

- Pick the tier the repo belongs to and match its floor.
- Set `requires-python`, the ruff `target-version`, and the CI matrix consistently (drop
  matrix rows below the floor).
- Bumping a **published** package's floor (anything on PyPI) is a compatibility change —
  ship it with a version bump and a release, not silently.
