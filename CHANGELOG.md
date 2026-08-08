# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **The docs site covers `validation.monitor` where a reader would look for it.** #121
  fixed the README; the site had the same hole in three more places. The homepage's "What
  crucible answers" grid enumerated eight capabilities and stopped at the gauntlet, the
  run-modes page listed five modes and stopped there too, and the visualization catalog
  documented every public `report` panel except `monitor_panel`, which was the only one
  missing.

  On the run-modes page the monitor is deliberately **not** a sixth row of the ladder.
  That page's frame is escalating strictness over one fixed log, and the monitor is a peer
  of the gauntlet running afterwards against a different log, so it sits below a break
  with a line saying why. Listing it as the strictest rung would teach exactly the
  misreading `docs/edge_monitor.md` settles in the negative, and the operational cost of
  that misreading is real: running the monitor before a promotion has frozen a baseline is
  what makes a decay trigger fire every cycle and inflate the multiple-testing denominator
  of the next verdict.

  `docs/gen_figures.py` grows a `panel_monitor_panel` figure. It is the one panel no
  single log can feed, needing a frozen baseline plus a separate live log, so it is drawn
  from the §14 example rather than the Donchian run. It deliberately shows the **healthy**
  live book, where the trailing read crosses the soft SLIPPING line on a book whose edge
  never decayed while the CUSUM below peaks near 29σ against a 35.1σ boundary and stays
  silent. That is the panel's argument; a picture of the halved book would show a bigger
  alarm and teach less.
- **The README covers `validation.monitor`.** It enumerated every other subpackage's API
  with a worked block and had zero mentions of `edge_monitor`, `EdgeBaseline` or the
  monitor at all, despite that module being the headline of two releases. It now gets a
  section beside the others: the freeze-at-promotion snippet, the three channels, and the
  three properties that are the point (only the calibrated channel escalates, the baseline
  cannot be rebuilt, it anchors to a search-corrected number), plus the standing caveat
  that it has never met real decay. `deflated_expectancy` is named in the validation
  section, which also predated it.

  `docs/edge_monitor.md` is down to one open item as a result, and says so rather than
  leaving the section looking longer than it is.

## [0.7.0] - 2026-08-03

A monitoring-accuracy release. Both entries concern the firing-rate channel, and
together they change it from a rule that could not fire into one that can, while
recording why it still may not escalate.

The channel compares the live firing rate against the baseline's. Anchoring that
baseline to the validated log's whole span made it insensitive on any book whose rate
has trended, which is the fix. Trying to give the same channel a calibrated detector so
it could return DEGRADED did not work, and the measurements that killed it are written
down so the attempt is not repeated.

### Changed
- **`docs/edge_monitor.md` records why the firing-rate channel stays uncalibrated**, and
  drops the claim that it could be. That item sat at the top of the page's open list from
  the day the monitor shipped, on the reasoning that trade arrivals are approximately
  Poisson and an arrival-process test would therefore give the channel a stated
  false-alarm rate. The reasoning was wrong and the claim is removed rather than softened.

  Three detectors were built and measured against a real book's arrivals, each designed
  for a 10-year false-alarm budget: an exponential CUSUM on inter-arrival gaps at three
  window lengths (0.36x / 0.35x / **0.18x** of the stated budget) and a Poisson CUSUM on
  monthly counts (0.32x). All fire 3x to 5x more often than advertised, and unlike the
  expectancy CUSUM's 1.39x skew inflation the error spends margin rather than buying it.

  Calibrating the boundary empirically removes the bias but not the problem. Solved
  against the real counts it leaves the delivered budget spanning **5.3 to 77.9 years**
  for a stated 10. The limit is the data, not the model: a book firing 34 trades/yr gives
  82 monthly periods in a 7-year window, and CUSUM run length depends on a tail that 82
  samples cannot pin down.

  `tests/test_frequency_calibration.py` pins the two general claims synthetically (80
  periods leave an 8.6x spread in the delivered budget where 2,000 leave 1.3x), so the
  limit is re-derivable without the book, and asserts that only the calibrated CUSUM can
  still reach `DEGRADED`.

### Added
- **`EdgeBaseline.from_log(..., rate_window_years=)`**, measuring the baseline firing rate
  over the last N years of the validated log rather than its whole span, and recording the
  window on `EdgeBaseline.rate_window_years` so a verdict can say which claim it is making.

  The default is unchanged (whole span), so nothing moves silently. But the whole span is
  the wrong anchor for any book whose firing rate has trended, and the failure is quiet:
  the full-span mean sits below what the book currently does, so the opportunity-set
  channel compares live behaviour against a rate the book left behind years ago and reads
  healthy while the rate falls.

  Measured on a real 40-year trend book: **23.4/yr over the full span against 33.9/yr over
  its last 7 years**. Anchored to the full span, a collapse to 30% below current behaviour
  still scored 1.02 and could never trip the 0.6 line. The channel covering the one failure
  the other two cannot see was structurally desensitised, not merely uncalibrated.

  It is a windowed measurement taken ONCE, at promotion, not a rolling one. `edge_monitor`
  still has no parameter from which a baseline or its rate could be rebuilt, which
  `test_the_window_never_re_measures_after_promotion` pins.

## [0.6.0] - 2026-08-02

The release that makes the monitor's baseline defensible and its calibration mean
something. 0.5.0 shipped the edge monitor the same day; running it against a real
47-market book rather than the synthetic fixtures it was built on immediately exposed
three calibration problems and confirmed the one gap already known, so this follows
closely and deliberately.

What changed, in order of how much it matters if you are already using 0.5.0:

1. **`deflated_expectancy` exists.** 0.5.0's `EdgeBaseline` accepted a deflated number and
   nothing in the package produced one, so every real baseline was the raw in-sample mean.
   The monitor's central argument, that decay should be measured from a search-corrected
   reference, was a docstring. It is now a function.
2. **The false-alarm budget is denominated in years, not trades.** If you built a design
   under 0.5.0 against a baseline that knows its firing rate, its calibration changes here,
   and for a slow book it changes a lot: the old default gave a 23-trades/yr book a nominal
   detection latency of 24 years.
3. **The opportunity-set channel works for logs without `entry_date`.** Under 0.5.0 it
   silently switched itself off for a whole ordinary class of books.


### Added
- **`validation.deflated_expectancy`**, closing the gap that stood at the top of
  `docs/edge_monitor.md`'s open list since the monitor shipped. `EdgeBaseline` took a
  `deflated_expectancy` float and nothing in the package produced one: `deflated_sharpe`
  corrects a Sharpe and returns a **probability**, which is the right output for a gate
  and useless to a monitor. A monitor needs a number in R to anchor to, and without one
  the only available anchor was the sample mean, which is the number the parameters were
  optimized on.

  The conversion rides on the bar `deflated_sharpe` already uses: `SR0`, the expected
  maximum per-trade Sharpe of N noise trials, carried back into R by the winner's own
  sigma and subtracted, `deflated = mu - sigma * SR0`. Both now call one
  `_expected_max_sharpe`, so two corrections for one search cannot disagree about how big
  the search was. `EdgeBaseline.from_log` accepts the `DeflatedExpectancy` object as well
  as a float, because handing over the wrong field of a result you already computed
  anchors the monitor to the pre-correction number while reporting `deflated=True`.

  **It takes trial LOGS, not trial Sharpes**, unlike `deflated_sharpe`. The Sharpes are
  computed inside so their clock cannot be got wrong: multiplying a per-month Sharpe by a
  per-trade sigma yields a haircut in no units at all, silently, which is the v0.4.0 units
  bug in a new costume.

  **It is a bias correction, not a significance test**, and the docstring, the `__str__`
  and the property name all say so. It removes the selection bias a search of this size is
  *expected* to produce, so a pure-noise winner still clears zero roughly half the time
  (measured at 56% / 47% / 44% for N = 5 / 20 / 100, against `deflated_sharpe` correctly
  calling 0% of the same draws significant). The result's property is `is_positive` rather
  than `survives` for exactly that reason. A correction that leaves nothing raises when it
  reaches `EdgeBaseline`, whose existing refusal now names deflation as a cause.

### Changed
- **`examples/edge_monitor.py` now runs a real 64-config search** and deflates the winner,
  where it previously applied a hardcoded `raw_mean * 0.8` under a comment beginning
  "Pretend". Every figure in tutorial §14 moved as a result (the haircut is 26% of the raw
  edge, not 20%, and the naive baseline flatters by 36%, not 25%). The example now exposes
  `promoted_book()` so the tutorial's numbers have one source.

- **The CUSUM's false-alarm budget is now stated in CALENDAR TIME.** New
  `Thresholds.monitor_arl0_years` (default 25), converted using the baseline's own firing
  rate; `monitor_arl0_trades` stays as the fallback when the rate is unknown.
  `CusumDesign` reports `arl0_basis` and renders both ARLs in years.

  A budget in trades silently means different things to different books. 7,500 trades is
  about 50 years at 150 trades/yr and about 320 at 23, and the slower book was getting a
  detector whose nominal detection latency was 24 years, which is not a monitor. Calendar
  time is the unit the decision is actually made in, so one default now means one thing.

  **Behaviour change** for any baseline that knows its firing rate: the design gets
  tighter and detection faster. The worked example moves from a 7,500-trade budget to
  3,752 trades (25.0 years), and its detection latency from 1,099 trades to 806 (5.4
  years). Tutorial §14 and `tests/test_edge_monitor_example.py` are updated together.
- **Corrected the module's claim about how far the Gaussian ARL approximation holds.** It
  said the nominal figure stays "within 0.96x to 1.17x", generalizing from a synthetic
  10%-win-rate book whose skew is +3. A real pooled trend-following book runs about skew
  +5, with single trades near +39R against losses capped near -1R, and measures ~2.0x. The
  error grows with skew and with the boundary: ~1.5x at skew +4, ~2.6x at +8, ~4.0x at
  +11, and worse at larger `monitor_arl0_trades`.

  The drift is conservative and costs nothing: measured, the inflation does NOT carry over
  to `arl1`, which tracked nominal within a few percent at every budget tested. In control
  the statistic hovers near zero and alarms only via a rare large excursion, exactly where
  a fat tail bites; under a real shift it reaches the boundary by drift, where tail shape
  barely matters. So a skewed book buys extra false-alarm margin for free. Both the
  docstring and [docs/edge_monitor.md](docs/edge_monitor.md) now carry the graded table,
  and two tests pin it so the claim cannot quietly drift back.

### Fixed
- **`tests/test_edge_monitor_example.py` rebuilt the baseline it was supposed to pin**,
  instead of importing it. So the "reproducibility guard" for tutorial §14 passed
  unchanged while the example it guards printed entirely different numbers. It now imports
  `promoted_book()`. A guard that reconstructs what it guards is not a guard.
- **The opportunity-set channel switched itself off for any log without `entry_date`.**
  `_trades_per_year` read `entry_date` only, so a log built from a return column and an
  exit date, which is an ordinary shape, produced `trades_per_year=None` on the baseline
  and `frequency_ratio=None` in every verdict. It said so in `reasons` and nothing was
  wrong-but-silent, yet the effect was that the one channel covering a signal that stops
  firing was dead by default for a whole class of books. It now falls back to `exit_date`
  (a rate is `n / span`, and either column dates the same trades closely enough), with
  `entry_date` still preferred when both are present.

## [0.5.0] - 2026-08-02

The edge-monitor release. The gauntlet asks "is this edge real?" once, over a fixed log;
this adds the question that follows a promotion, *is it still real?*

Also ships everything 0.4.0 carried: **0.4.0 was version-bumped and changelogged on
2026-07-28 but never tagged or published**, so its detrended-null units fix reaches PyPI
here for the first time. Upgrading from 0.3.1 means reading the 0.4.0 section below as
part of this release.

### Added
- **`validation.monitor`**: the post-promotion counterpart to the gauntlet. `EdgeBaseline`
  (frozen at promotion), `cusum_design`, `edge_monitor` returning
  HOLDING / SLIPPING / DEGRADED, plus `rolling_expectancy`, `cusum_path` and
  `empirical_arl`. Stateless and capital-free like the rest of the package: it owns no
  clock, persists nothing, and emits a verdict rather than a sizing action. Five
  `Thresholds` entries (`monitor_detect_shift`, `monitor_arl0_trades`, `monitor_window`,
  `monitor_slip_ratio`, `monitor_min_frequency_ratio`).

  Three properties are structural rather than advisory, each with a test:
  - **Only the calibrated detector may say DEGRADED.** The CUSUM has a stated false-alarm
    rate, so it alone escalates. The rolling-window ratio and the firing-rate ratio have
    no such calibration and cap out at SLIPPING. On a book whose edge never decayed and
    sits a quarter *above* baseline, the 200-trade trailing read still swings between
    -31% and 240% of baseline; letting a rule like that govern sizing cuts healthy books.
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
  carries the aggregate.
- **`report.monitor_panel`**: the decay panel. Trailing expectancy against the frozen
  baseline and the SLIPPING line on top, the CUSUM against its alarm boundary underneath
  with the first crossing marked. The two rows are the point: the top one wanders far
  enough to cross the soft line on a book that never decayed, the bottom one carries a
  stated false-alarm rate. Behind the `[report]` extra like every other plotly block, and
  capital-free (R and sigma units, never an account).
- **Tutorial §14 and [`examples/edge_monitor.py`](examples/edge_monitor.py)**: the monitor
  read end to end, seeded and synthetic. Freezing a deflated baseline, designing the
  detector from a stated false-alarm budget, checking its Gaussian ARLs against the book's
  own returns, and three live books (edge intact / edge halved / signal drying up) against
  one frozen baseline. Every number the tutorial quotes is pinned by
  `tests/test_edge_monitor_example.py`, matching the §13 convention.
- **[docs/edge_monitor.md](docs/edge_monitor.md)**, the design note. Records what shipping
  settled (the monitor belongs in crucible; it is a peer of the gauntlet rather than a
  fifth gate) and what is still open, chiefly that nothing yet deflates an expectancy and
  nothing outside the tests calls the module.

### Fixed
- `report.monitor_panel` pins its CUSUM y-axis to include the alarm boundary. Letting
  plotly autoscale to the data hid `h` on exactly the healthy books where the useful
  reading is how much room is left: on the default design a healthy book peaks near 25
  sigma against an `h` of 47, so the boundary fell outside the frame.

## [0.4.0] - 2026-07-28

**Never published.** Bumped in `pyproject.toml` and documented here, but no `v0.4.0` tag
was ever pushed, so PyPI went 0.3.1 -> 0.5.0. The changes below shipped in 0.5.0.

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
