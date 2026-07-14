# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/mspinola/crucible/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mspinola/crucible/releases/tag/v0.1.0
