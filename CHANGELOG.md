# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `report.monitor_panel`: the decay panel. Trailing expectancy against the frozen
  baseline and the SLIPPING line on top, the CUSUM against its alarm boundary
  underneath, with the first crossing marked. The two rows are the point: the top one
  wanders far enough to cross the soft line on a book that never decayed, the bottom one
  carries a stated false-alarm rate. Behind the `[report]` extra like every other plotly
  block, and capital-free (R and sigma units, never an account).
- `validation.cusum_path`: the detector's running statistic as a `pd.Series`, one value
  per live trade. The series `edge_monitor` reduces to a verdict, and what the panel
  plots. Like `edge_monitor` it takes a frozen `EdgeBaseline` and has no parameter from
  which one could be rebuilt. `edge_monitor` now derives its alarm index from this same
  path rather than a second inline loop.

### Fixed
- `report.monitor_panel` pinned its CUSUM y-axis to include the alarm boundary. Letting
  plotly autoscale to the data hid `h` on exactly the healthy books where the useful
  reading is how much room is left: on the default design a healthy book peaks near 25
  sigma against an `h` of 47, so the boundary fell outside the frame. Caught before
  release by checking the plotted ranges rather than only asserting on the HTML.
- Tutorial §14 and [`examples/edge_monitor.py`](examples/edge_monitor.py): the edge monitor
  read end to end, seeded and synthetic. Freezing a deflated baseline, designing the
  detector from a stated false-alarm budget, checking its Gaussian ARLs against the book's
  own returns, and three live books (edge intact / edge halved / signal drying up) against
  one frozen baseline. Every number the tutorial quotes is pinned by
  `tests/test_edge_monitor_example.py`, matching the §13 convention.

  The example exists mainly to make one claim reproducible: on a book whose edge never
  decayed and is a quarter *above* baseline, the 200-trade trailing read swings between
  -31% and 240% of baseline and dips under the 50% line in 9% of windows, while the CUSUM
  peaks at 53% of threshold and never fires. The design note previously asserted a similar
  figure from an uncommitted scratch run; it now cites the example.

- `validation.monitor`: the post-promotion counterpart to the gauntlet. The gauntlet asks
  "is this edge real?" once, over a fixed log; this asks whether it is *still* real.
  `EdgeBaseline` (frozen at promotion), `cusum_design`, `edge_monitor` →
  HOLDING / SLIPPING / DEGRADED, plus `rolling_expectancy` and `empirical_arl`. Stateless
  and capital-free like the rest of the package: it owns no clock, persists nothing, and
  emits a verdict rather than a sizing action. Five `Thresholds` entries
  (`monitor_detect_shift`, `monitor_arl0_trades`, `monitor_window`, `monitor_slip_ratio`,
  `monitor_min_frequency_ratio`).

  Three properties are structural rather than advisory, each with a test:
  - **Only the calibrated detector may say DEGRADED.** The CUSUM has a stated false-alarm
    rate, so it alone escalates. The rolling-window ratio and the firing-rate ratio have
    no such calibration and cap out at SLIPPING. A trailing read printed "59% of baseline"
    on a book whose true edge was intact and slightly *above* baseline; letting a rule
    like that govern sizing cuts healthy books.
  - **No re-baselining.** `edge_monitor` has no parameter from which a baseline could be
    rebuilt, since a baseline recomputed from current data re-fits onto the drifted
    reality and can never fire.
  - **An undeflated baseline is allowed but never silent.** `EdgeBaseline.deflated` rides
    in the verdict, on the same principle as `variant_count()` refusing a typed-in int.

  CUSUM parameters are derived from a stated shift and a stated false-alarm budget, not
  typed in; `arl1` reports the resulting detection latency as an up-front cost.
  `arl0`/`arl1` are **means** of a strongly right-skewed distribution, and `empirical_arl`
  returns mean and median alongside a resampling of your own returns, because the Gaussian
  design assumption is checkable and quoting a median against a mean misstates latency by
  roughly a third. Measured, that assumption holds better than expected: empirical ARL0
  stays within 0.96x to 1.17x of nominal from an ordinary 43%-win-rate shape out to a
  lottery-shaped 10%-win-rate book, because the boundary sits many sigma away and the CLT
  carries the aggregate. See [docs/edge_monitor.md](docs/edge_monitor.md), which records
  both what merging settled (the monitor belongs in crucible; it is a peer of the
  gauntlet, not a fifth gate) and what is still open, chiefly that nothing yet deflates
  an expectancy and nothing outside the tests calls the module.

## [0.4.0] - 2026-07-28

### Added
- `detrended_timing_null(..., scale=...)` and `gate_real(..., null_scale=...)` /
  `run_gauntlet(..., null_scale=...)`: per-trade multipliers that denominate the detrended
  timing null in the trade log's own return unit. The null draws simple (fractional)
  returns; a log denominated in R (1R = entry-to-stop risk) is a different unit, and since
  `R = fractional_return * (entry / risk)` exactly, passing `scale = entry / risk` puts the
  null in R. Backward compatible: `scale=None` (the default) is the previous behaviour,
  correct for a simple-return log.

### Fixed
- The detrended `beats_random_timing` check was comparing an observed expectancy in R
  against a null in fractional returns, off by the ~`entry/risk` factor (tens to hundreds),
  so it passed almost any positive-expectancy book. Callers with an R-denominated log should
  now pass `null_scale`. See the npf finding `docs/methodology/detrended_null_r_units.md`.

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
