# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

- **[AGENTS.md](AGENTS.md)** — the contributor contract (public-API stability, hard checks,
  determinism, capital-free). Treat it as constraints, not suggestions; propose edits to a
  maintainer rather than expanding it.
- **[docs/architecture.md](docs/architecture.md)** — the module map, the signal→verdict
  spine, the data structures, and a "where to make changes" table. Start there for anything
  non-trivial.

## Commands

```bash
pip install -e ".[dev,report]"   # what CI installs; report extra so tearsheet tests run
python -m ruff check src tests   # lint (ruff is pinned to 0.15.22 in the dev extra)
python -m pytest -q              # full suite
python -m pytest tests/test_gauntlet.py::test_real_fails_on_no_edge -q    # one test
python examples/quickstart.py    # CI's smoke test — synthetic data, no network
```

Docs (MkDocs Material):

```bash
pip install -r requirements-docs.txt && mkdocs serve
```

CI publishes with `mkdocs build --strict`, so a broken link or nav entry fails the build.
`docs/gen_figures.py` (regenerates tutorial PNGs; needs the `report` extra + headless
Chrome) and `docs/gen_pdf.py` (tutorial PDF) are maintainer tooling, not part of
`mkdocs build`.

CI matrix is Python 3.10–3.12. The floor is fleet policy — see
[docs/python_support.md](docs/python_support.md); `requires-python`, ruff `target-version`,
and the CI matrix must move together.

### Running tests from a git worktree

The repo's `.venv` editable-installs crucible pointing at the **main checkout's** `src/`,
so a worktree's code is not what gets imported. Prefix with `PYTHONPATH` — a plain `.pth`
loses to `PYTHONPATH`:

```bash
PYTHONPATH=$PWD/src python -m pytest -q
```

## Architecture in one paragraph

Everything pivots on one artifact: `TradeLog` (`edge/trade_log.py`) — a frozen wrapper over
a DataFrame whose only required column is `r`, the return in **R-multiples** (1R = risk at
entry). `edge` *produces* it (`barrier_trades`) and describes it (`edge_report`,
`reality_check` → HELD/FRAGILE/FAIL with a bootstrap CI and p-value). Every other module
*judges* it: `validation` (holdout, walk-forward, permutation, PBO/deflated Sharpe, and the
`run_gauntlet` gate), `breadth` (how many independent bets a correlated book holds), `ml`
(the same honesty aimed at a model's scores rather than a trade log), `report` (self-
contained plotly HTML). The gauntlet runs REAL → STRONG → DURABLE → GENERAL, and a
`Gate`/`Gauntlet` verdict is the AND of its **hard** checks with no setter and no override
flag — that un-overridability is the product.

## Invariants that CI actually enforces

`tests/test_boundaries.py` asserts the import surface with `ast`:

- `edge`, `validation`, `breadth` import **only** numpy + pandas; `ml` and `report` may add
  plotly. No scipy, sklearn, or xgboost in the core (`pbo.py` uses stdlib
  `statistics.NormalDist`; Spearman is done via ranks).
- Banned outright: vectorbt/vectorbtpro (the engine lives outside crucible behind a seam),
  `npf` (dependency runs one direction only), schedulers, and durable-state libs
  (sqlite3, pickle, sqlalchemy, redis — a lens persists nothing).
- The packaging surface is checked against `pyproject.toml`, so a new module cannot quietly
  widen what ships.

Not machine-enforced but equally load-bearing: **capital-free** (no position sizing, equity
curve, drawdown, or CAGR — those live downstream) and **deterministic** (same input, same
seed, same verdict; bootstrap/permutation take an explicit seed).

## Conventions

- Public symbols are the contract. Each subpackage's `__init__.py` declares `__all__`; don't
  rename, move, or drop one without a deprecation path.
- Gauntlet numbers live in `Thresholds` (`validation/thresholds.py`), never inline. Retuning
  means changing that dataclass in the open with a rationale, not adding an option.
- Two things are named "gauntlet": `Gauntlet` the result class (`validation/gate.py`) and
  `gauntlet.py` the module holding `run_gauntlet` + the four gate factories.
- Data-mining corrections take the honest N from a `SearchSpaceLog` ledger, not a typed-in
  int. `variant_count()` is the helper both routes through.
- Tests use the synthetic `ohlc` fixture in `tests/conftest.py` (seeded RNG, ~6y of daily
  bars) — no network in the suite. `examples/real_data_yfinance.py` is the only networked
  example.
- Public behavior changes update `docs/` and `CHANGELOG.md` in the same change. Every claim
  in the tutorial is expected to run.

## Releasing

Bump `version` in `pyproject.toml`, move the `CHANGELOG.md` entry out of *Unreleased*, then
tag `vX.Y.Z`. The release workflow **fails if the tag doesn't match the pyproject version**.
Publishing is PyPI Trusted Publishing (OIDC, no stored tokens); there is no TestPyPI dry run
(that name belongs to an unrelated project). Versions start at 0.2.0 — an unrelated 2011
package holds 0.1.0 on the `crucible` PyPI name.
