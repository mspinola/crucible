# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] — 2026-07-23

### Changed
- Raised the minimum Python to 3.10 (`requires-python = ">=3.10"`), matching the fleet
  library-tier floor policy. See `docs/python_support.md`. NumPy/pandas already require
  3.10+; no code changes.

## [0.3.0] — 2026-07-22

The honest-N release. 0.2.0 introduced `SearchSpaceLog` as the ledger of every variant a
search actually tried, and then never handed that number to the corrections that needed it.
`deflated_sharpe` derived N from the number of configs it was given scores for, which is
the count you remember rather than the count you ran. This release closes that gap.

The practical effect is that corrections get stronger, and results that passed may stop
passing. In the strategy repo that consumes this library, pricing a 45-market scan against
its full 129-variant search space moved two apparent survivors from 98% and 99% deflated
Sharpe to 0% and 2%, and took the number of books clearing the gate from two to zero. That
is the correction working, not a regression.

### Added
- **`deflated_sharpe(..., n_trials=)`**, so the honest denominator can be supplied rather
  than inferred. Accepts an `int` or a `SearchSpaceLog` directly. Omitted, it falls back to
  the previous behaviour of counting the scores it was given, so existing calls are
  unaffected.
- **`variant_count(n_variants)`**, the small public helper both corrections now route
  through. Takes an `int` or a `SearchSpaceLog` and returns the count, so a caller cannot
  drift from the ledger by retyping a number.
- Import-boundary tests in CI. `crucible.edge` and `crucible.validation` are held to
  numpy/pandas, with no vectorbt, no orchestration or state, and no dependency on any
  strategy package. The packaging surface is checked against `pyproject.toml` in the same
  pass, so a new module cannot quietly widen what ships.

### Changed
- **`sidak_correction(p_raw, n_variants)`** and **`run_gauntlet(..., n_variants=)`** now
  accept a `SearchSpaceLog` as well as an `int`. Prefer passing the ledger: it counts every
  variant tried, including the ones that errored or scored nothing, which is the honest
  denominator and the one a caller is least motivated to inflate.
- Separated *how many configs were tried* from *how many were scored* inside
  `deflated_sharpe`. Those were the same variable and are not the same quantity, which is
  what made the ledger decorative.

### Notes
- Backward compatible. Every new parameter is optional and defaults to the 0.2.0 behaviour.
- The `crucible-quant` redirect shim under `packaging/` points the old distribution name at
  this one. It is a separate distribution and is excluded from this package's sdist.

## [0.2.0] — 2026-07-21

The gauntlet release. 0.1.0 could describe an edge and test whether it was real. 0.2.0
runs the whole ladder as one audited verdict, prices the search that found the edge, and
reports the result as a composable tearsheet.

### Added
- **`crucible.validation.run_gauntlet`**, the ordered four-pillar verdict
  (REAL → STRONG → DURABLE → GENERAL), which passes only if every gate does. Built on
  new gate primitives (`Gate`, `GateCheck`, `Gauntlet`, `Thresholds`) and the individual
  stages `gate_real`, `gate_strong`, `gate_durable`, `gate_general`.
- **Pricing the search itself**, the answer to "how much did *selecting* this config
  overfit?":
  - `pbo_cscv` / `PBOResult`, Probability of Backtest Overfitting via CSCV.
  - `deflated_sharpe` / `DeflatedSharpe`, Sharpe corrected for multiple testing.
  - `SearchSpaceLog`, an honest N for that correction, so it counts every variant you
    actually tried rather than the ones you remember.
  - SELECT/overfit bars on `Thresholds` (`max_pbo`, `min_deflated_sharpe`).
- **`crucible.breadth`**, capital-free effective-N: `effective_n`,
  `participation_ratio`, and the `Breadth` result. How many *independent* bets a
  correlated book really holds, read off the correlation eigenvalues, with no equity
  curve anywhere in the calculation.
- **`crucible.ml`**, the walk-forward ML path. Still numpy/pandas only, it imports
  neither scikit-learn nor xgboost:
  - `information_coefficient`, `fold_ic`, and `alpha_gate` / `AlphaGateError` for
    signal strength.
  - `quantile_decay`, `decay_tearsheet`, `score_by_outcome`, and `DecayTable` for how
    fast a score decays across quantiles.
  - `redundancy_droplist`, `cramers_v`, and `RedundancyReport` for feature overlap.
  - `asof_window` and `window_before`, point-in-time windows that keep features honest.
- **Significance under serial dependence**: `block_bootstrap_pvalue` and
  `block_bootstrap_ci` resample contiguous blocks of an ordered period-return series
  (circular or stationary), so autocorrelation survives in the null. This is the honest
  p-value for a pooled multi-asset book, where the i.i.d. trade bootstrap treats
  trades as exchangeable and breaks their time clustering.
- **More nulls**: `detrended_timing_null`, and `spa_test` (Hansen's Superior Predictive
  Ability) alongside the existing White's Reality Check.
- **Out-of-sample shapes beyond a single split**: `segmented_holdout`,
  `windowed_segments`, and `full_sample`, with `SegmentedHoldout`, `WindowedSegments`,
  and `WindowCell`.
- **`bootstrap_metric_cis`**, confidence intervals across the whole metric set in one
  pass, and the walk-forward diagnostics `fold_dispersion` and
  `walk_forward_efficiency`.
- **`crucible.report` went from two exports to twenty.** `gauntlet_report` renders the
  full four-pillar page, and the pieces are composable and theme-aware, so a custom
  page can be assembled from `verdict_banner`, `verdict_summary`, `pillar_bullets`,
  `gate_block`, `edge_panels`, `metrics_table`, `title_lockup`, and `report_css`. New
  panels: `monthly_r`, `equity_drawdown`, `exit_reason_breakdown`, `holding_vs_r`,
  `exit_efficiency_dist`, `edge_ratio_curve`, `gross_net_equity`,
  `concurrency_timeline`, and `segment_forest`. A bare-embed mode drops the page chrome
  for embedding in a host document.
- **A published tutorial**, *From Trade Log to Verdict*, rendered with MkDocs Material
  at <https://mspinola.github.io/crucible/> and downloadable as a PDF. The site also
  carries a landing page, an Architecture page for contributors, and a visualization
  catalog.
- **`[data]` extra** (`cotdata`), the optional futures/COT companion.

### Changed
- **Distribution renamed from `crucible-quant` to `crucible`.** The import name is
  unchanged, `import crucible` works exactly as before, so no code changes are needed.
  Installers move from `pip install crucible-quant` to `pip install crucible`.
- **Releases start at 0.2.0, not 0.1.1.** The `crucible` project on PyPI already holds a
  0.1.0 from an unrelated 2011 package, and PyPI never allows a version to be reused.
  Versions at or below 0.1.0 are the historical `crucible-quant` line.
- `gate_durable` takes an optional SQN-WFE criterion, and `gate_real` an optional
  detrended null.
- A GENERAL-only miss now reports as a scope-limited verdict rather than an outright
  FAIL. The edge held, what is unproven is its generality.
- Tearsheets default to a "costs not attested" badge instead of leaving net numbers
  implied, and every tearsheet carries the logo lockup rather than just the gauntlet
  page.

### Fixed
- `sqn` now guards dispersion at the level of floating-point noise rather than only
  exact-zero standard deviation. A degenerate walk-forward fold of near-equal
  R-multiples could previously report an SQN around 1e14.
- `SearchSpaceLog.mark_selected` no longer double-counts the winning variant, which
  inflated the search-corrected N and so made the correction look harsher than it was.

## [0.1.0] — 2026-07-14

Initial release — the capital-free trading-edge evaluation core.

### Added
- **`crucible.edge`** — the capital-free core (numpy/pandas only):
  - `TradeLog` — the one schema everything speaks (`r` in R-multiples, plus
    optional `mfe` / `mae` / `bars_held` / `prob` / `entry_date` / `exit_date`).
  - Edge metrics — `expectancy`, `profit_factor`, `payoff_ratio`, `win_rate`,
    `sqn`, and the excursion family (`excursion_ratio`, `e_ratio`,
    `time_asymmetry`, `exit_efficiency`), assembled by `edge_report`.
  - Honesty layer — `bootstrap_ci`, `p_value_positive`, `reality_check`
    (HELD / FRAGILE / FAIL), and `random_entry_null`.
  - `barrier_trades` — a generic OHLC + entry-signal → `TradeLog` simulator, and
    `random_entries` for the null model.
- **`crucible.validation`** — does the edge survive out of sample:
  - `holdout` — leakage-controlled early-train / late-confirm split.
  - `walk_forward` — anchored/rolling Pardo walk-forward with per-fold
    Walk-Forward Efficiency, stitching OOS slices into one `TradeLog`.
  - `permutation` — `sign_permutation_pvalue`, `sidak_correction`, and
    `whites_reality_check` (max-statistic across every variant searched).
- **`crucible.report`** (behind the `[report]` extra) — `tearsheet()` writes a
  self-contained HTML page (verdict banner, metric scorecard, R-multiple
  distribution, cumulative R, MFE/MAE excursion, bootstrap expectancy), and
  `cumulative_r()`. Capital-free — charts summed R, never an equity curve.
- **`crucible.strategies`** — `ma_cross`, `macd_cross` example signals.
- Examples: `quickstart.py`, `validation.py`, `tearsheet.py` (synthetic, no
  network), and `real_data_yfinance.py` (real prices via the `[examples]` extra).
- CI across Python 3.9–3.12; tag-triggered PyPI release via Trusted Publishing.

[Unreleased]: https://github.com/mspinola/crucible/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/mspinola/crucible/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/mspinola/crucible/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mspinola/crucible/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mspinola/crucible/releases/tag/v0.1.0
