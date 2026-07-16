# crucible

[![PyPI](https://img.shields.io/pypi/v/crucible-quant)](https://pypi.org/project/crucible-quant/)
[![CI](https://github.com/mspinola/crucible/actions/workflows/ci.yml/badge.svg)](https://github.com/mspinola/crucible/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tutorial](https://img.shields.io/badge/docs-tutorial-1c6b73)](https://mspinola.github.io/crucible/)

**Measure the edge before you ever open a $100k account.**

Most trading "edges" are artifacts of a small sample. `crucible.edge` takes a
**trade log** and tells you — with a confidence interval and a p-value — whether
the edge is real. No account, no position sizing, no equity curve. It's the thing
you run *before* a backtester.

```bash
pip install crucible-quant              # core: metrics + stats + simulator (numpy/pandas only)
pip install "crucible-quant[examples]"  # + yfinance, to run the demo below on real data
```

> Installed as **`crucible-quant`**, imported as **`crucible`** (`import crucible`).

> 📖 **[Read the tutorial →](https://mspinola.github.io/crucible/)** — *From Trade Log to
> Verdict: the statistics of a significant edge* — every technique worked end to end on a
> Donchian breakout. ([Markdown source](docs/statistical_significance_tutorial.md).)

## What crucible answers

From a single trade log up to the whole correlated book, one question at a time —
each harder than the last, each answered out loud by a named function:

- ✅ **Describe the edge** — expectancy, profit factor, payoff, SQN, MFE/MAE efficiency → `edge_report`
- ✅ **Quantify sampling noise** — a bootstrap CI and p-value on every headline metric → `bootstrap_ci` · `p_value_positive`
- ✅ **Rule out data-mining luck** — sign-permutation p-value (Masters) + a random-entry reality check → `permutation` · `reality_check`
- ✅ **Rule out drift** — a leakage-controlled early-train / late-confirm split → `validation.holdout`
- ✅ **Confirm out-of-sample** — Pardo walk-forward with purge / embargo hygiene → `validation.walk_forward`
- ✅ **Price the search itself** — how much did *selecting* the best config overfit? Probability of Backtest Overfitting (CSCV) + deflated Sharpe → `validation.pbo`
- ✅ **Account for correlation** — the effective number of *independent* bets across a correlated book → `breadth.effective_n`
- ✅ **…then all of it, as one gate** — run every check above as an ordered, audited gauntlet (REAL → STRONG → DURABLE → GENERAL) that passes only if every gate does → `validation.run_gauntlet`

## 30-second example

```python
import yfinance as yf
from crucible.edge import barrier_trades, edge_report, reality_check
from crucible.strategies import ma_cross

px = yf.download("ES=F", start="2010-01-01")       # any OHLC frame works
entries = ma_cross(px, fast=20, slow=50)            # your signal: a boolean Series

trades = barrier_trades(px, entries, side="long",   # signal -> trade log (in R)
                        tp=2.0, sl=1.0, timeout=20)  # 2R target, 1R stop, 20-bar cap

print(edge_report(trades))                          # the full capital-free scorecard
print(reality_check(trades))                        # <-- the verdict
```

```
======================================================
 EDGE REPORT (capital-free)
======================================================
Trades              : 214
Win rate            : 38.3 %
------------------------------------------------------
Expectancy          : +0.081 R      [PASS]
Profit factor       : 1.34          [PASS]
Payoff ratio        : 2.16          [INFO]
SQN-100             : 1.72          [INFO]
------------------------------------------------------
Excursion ratio     : 1.28          [PASS]
======================================================

VERDICT (expectancy): +0.081 R   95% CI [-0.031, +0.196]
                     p(edge>0) = 0.071        ->  FRAGILE
  point positive, but the CI straddles zero — not distinguishable
  from noise at this sample size. Do NOT size it up.
```

That `FRAGILE` block is the whole point: a positive expectancy that a backtester
would have shown you as a rising equity curve is, at this sample size,
**indistinguishable from noise**. crucible says so out loud.

> Runnable versions live in [`examples/`](examples): `quickstart.py`,
> `validation.py`, and `breadth.py` use synthetic data (no network);
> `real_data_yfinance.py` pulls
> real prices from Yahoo Finance (`pip install "crucible-quant[examples]"`) and runs the
> full pipeline — try `python examples/real_data_yfinance.py --ticker QQQ`.

## What's in the box

- **`TradeLog`** — one documented schema (`r` in R-multiples, plus optional
  `mfe` / `mae` / `bars_held` / `prob`). Everything speaks it.
- **Edge metrics** — expectancy, profit factor, payoff ratio, SQN, and the
  excursion family (MFE/MAE efficiency, E-ratio, time asymmetry, exit efficiency).
- **The honesty layer** — `bootstrap_ci`, `p_value_positive`, `reality_check`
  (HELD / FRAGILE / FAIL), and `random_entry_null` (did your signal beat
  coin-flip timing on the same prices?).
- **Book-level breadth** — `breadth.effective_n`: how many *independent* bets a
  correlated set of return streams really holds (N_eff, the participation ratio of
  the correlation eigenvalues) — the honest denominator for significance. Still
  capital-free: correlation structure only, no equity curve.

  ```python
  >>> from crucible.breadth import effective_n
  >>> effective_n(returns).n_eff     # 12-market book: 3 correlated blocs + a lone metal
  3.8                                # ...so it's really ~4 independent bets
  ```
  See [`examples/breadth.py`](examples/breadth.py) for the full factor breakdown.

- **A generic barrier simulator** — `barrier_trades`: OHLC + a boolean entry
  signal → a `TradeLog`. No instrument specifics.
- **Example signals** — `ma_cross`, `macd_cross`. Demos, not endorsed edges.

## Does the edge survive out of sample? — `crucible.validation`

The pooled reality check tells you if an edge is real *on the whole history*.
`crucible.validation` asks the harder question — does it hold on data the analysis
never touched?

```python
from crucible.validation import holdout, walk_forward, sign_permutation_pvalue

# 1. Early-train / late-confirm — leakage-controlled temporal split
print(holdout(trades, "2019-01-01", embargo_weeks=8))    # verdict is the LATE period

# 2. Sign-permutation p-value (Masters) — could the edge come from noise?
print(sign_permutation_pvalue(trades))

# 3. Pardo walk-forward — optimize params in-sample, confirm out-of-sample, stitch
wf = walk_forward(px, ma_cross, param_grid={"fast": [10, 20], "slow": [50, 100]},
                  is_days=365 * 3, oos_days=365)
print(wf)                          # per-fold IS->OOS efficiency (WFE)
print(reality_check(wf.stitched))  # the stitched-OOS verdict — the honest one
```

Also here: `sidak_correction` and `whites_reality_check` (max-statistic across every
variant you searched) for when a grid search flatters the best result — and, one
step further, `pbo_cscv` + `deflated_sharpe` (`crucible.validation.pbo`), which ask
how much the *act of selecting* the best config overfit: the Probability of Backtest
Overfitting (Bailey/López de Prado CSCV) over a trial matrix, and the Sharpe deflated
for the number of trials and its own skew/kurtosis.
See [`examples/validation.py`](examples/validation.py).

## One verdict for the whole edge — `crucible.validation.run_gauntlet`

The individual tools above each answer one question. The **gauntlet** runs them as an
ordered set of hard gates and returns a single audited pass/fail — the honest
scorecard, capital-free:

```python
from crucible.validation import run_gauntlet

gauntlet = run_gauntlet(
    wf.stitched,        # the honest log — stitched out-of-sample
    prices=px,          # enables REAL's random-timing null
    wf=wf,              # adds the DURABLE gate
    n_variants=4,       # size of your search -> REAL's Šidák correction
)
print(gauntlet.audit_report())
print(gauntlet.passed)  # True only if every gate that ran passed
```

The gates — **REAL** (not noise, corrected for the search) → **STRONG** (real at the
CI lower bound) → **DURABLE** (holds out-of-sample) → **GENERAL** (travels across
markets) — with two bring-your-own preambles (**DECLARE**, **CLEAN**) and a deliberate
handoff (**SURVIVE**: capital survivability is out of scope). Thresholds live in one
overridable `Thresholds`. Full write-up in [`docs/edge_gate.md`](docs/edge_gate.md).

## A shareable tearsheet — `crucible.report`

```bash
pip install "crucible-quant[report]"
```

```python
from crucible.report import tearsheet
tearsheet(trades, "sheet.html", title="SPY — 20/50 MA cross")
```

Writes a **self-contained** HTML page (plotly.js inlined, renders offline): the
verdict banner, the metric scorecard, the R-multiple distribution, cumulative R,
MFE/MAE excursion, and the bootstrap expectancy distribution behind the CI. Still
capital-free — it charts summed **R**, never an equity curve. See
[`examples/tearsheet.py`](examples/tearsheet.py).

## Is the ML signal real? — `crucible.ml`

The same honesty question, aimed at a model's **scores** instead of a trade log:
does a predicted probability actually rank outcomes, or is it noise, leakage, or a
redundant feature wearing a new name? Model-agnostic and capital-free — the core is
numpy/pandas (no sklearn/xgboost); only the tearsheet needs the `report` extra.

```python
from crucible.ml import (
    information_coefficient, alpha_gate,   # predictive power, + a PASS/FAIL gate
    quantile_decay, decay_tearsheet,       # does a higher score mean a better outcome?
    fold_ic, redundancy_droplist,          # out-of-fold IC; which features overlap
    asof_window,                           # a point-in-time slice that can't peek ahead
)

# `preds`: a frame with a continuous `score` and the realized `label` (+1 win / -1 or 0 loss)
ic = information_coefficient(preds)        # Spearman rank IC of score vs outcome
alpha_gate(ic, min_ic=0.03)                # raises AlphaGateError below the bar

decay = quantile_decay(preds)              # realized win rate per score quintile
print(decay.monotonic, decay.spread)       # a real edge climbs Q1 -> Q5

rep = redundancy_droplist(panel, features, target="fwd")
print(rep.kept, rep.dropped)               # keep the highest-|IC| of each redundant cluster
```

- **`information_coefficient` / `alpha_gate`** — Spearman rank IC of a score against
  its realized label, plus a PASS/FAIL gate to stop an edge-less or leaking model early.
- **`quantile_decay` / `decay_tearsheet`** — realized win rate per score quantile (a
  genuine edge rises monotonically Q1→Q5); the tearsheet renders it as self-contained HTML.
- **`fold_ic` / `redundancy_droplist`** — per-feature out-of-fold IC, and a
  keep-highest-|IC| drop-list that groups features by |Spearman| / Cramér's V.
- **`asof_window` / `window_before`** — leakage-safe point-in-time slices, so a live
  feature is built identically to its training twin.

Still capital-free: it judges *signals and features*, never an equity curve.

## What this is — and isn't

✅ Trade-level edge metrics, excursion efficiency, bootstrap CIs, a random-entry
reality check — all **capital-free**.

❌ No capital, position sizing, commissions, CAGR, drawdown, or
Monte-Carlo-on-equity. *If you want an equity curve, hand the `TradeLog` to
[quantstats](https://github.com/ranaroussi/quantstats). crucible stops at the edge.*

## How this compares to RealTest

[RealTest](https://mhptrading.com/) (Marsten Parker / MHP Trading) is a mature,
paid, portfolio-level backtester. crucible is not a competitor to it — it's the
step *before* it. If there's overlap worth being honest about, here it is.

**Where RealTest is strictly broader.** RealTest runs a real capital simulation:
position sizing, commissions, slippage, portfolio-level allocation, and the
equity-curve statistics that follow from it — CAGR/CAR, MaxDD, Sharpe, Sortino,
exposure. crucible has none of that on purpose. It stops at the trade log. If you
want an equity curve, hand the `TradeLog` to quantstats.

**Where the methodologies genuinely differ.**

- *Out-of-sample boundary handling.* Both tools do OOS/holdout splits and
  walk-forward with re-optimization. The difference is leakage control at the
  boundary. crucible's `holdout` requires a training trade to have **entered and
  exited** before the split (a trade whose window straddles the boundary can't
  leak into the fitted side) and drops an `embargo_weeks` band at the start of the
  test period. `walk_forward` adds the Pardo `purge_days` / `embargo_days` hygiene
  per fold. RealTest's documentation describes no equivalent purge/embargo — an
  open position simply carries across the boundary.

- *Walk-forward efficiency.* crucible reports a named Pardo WFE per fold
  (annualized OOS return / annualized IS return) and a mean across folds. RealTest
  gives you the out-of-sample equity to inspect but does not output a named
  efficiency ratio.

- *Monte Carlo.* Both use bootstrap (random selection **with replacement**).
  They resample different things for different questions. RealTest resamples
  trades or daily account changes to draw percentile bands on the **equity curve
  and drawdown** — "how bad could the path get?" crucible resamples the trade
  returns to put a confidence interval and a p-value on the **edge metric itself**
  (expectancy, profit factor, SQN) — "is the edge distinguishable from zero?"
  crucible does not produce equity/drawdown bands; RealTest does not produce a
  CI on expectancy.

**Reported statistics.** Expectancy, profit factor, payoff (RR), and MFE/MAE
exist in both. crucible additionally reports SQN natively (RealTest has no
built-in SQN statistic; it would take a custom formula) and wraps every headline
metric in a bootstrap CI + p-value, which RealTest does not.

**The actual reason to reach for crucible:** it's open-source (MIT), imports as a
Python package, needs no account or capital model, and is free. It answers one
narrow question — *is this trade-level edge real, or a small-sample artifact?* —
and answers it out loud, before you ever open a funded account or a full
backtester.

### Feeding crucible from a RealTest export

The two tools compose cleanly. RealTest runs headless from the command line
(`realtest -test script.rts`) and, with `SaveTradesAs` set, writes the trade log
to CSV. RealTest's default trade export carries these columns:

```
Trade, Strategy, Symbol, Side, DateIn, TimeIn, QtyIn, PriceIn, DateOut, TimeOut,
QtyOut, PriceOut, Reason, Bars, PctGain, Profit, PctMFE, PctMAE, Fraction, Size, Dividends
```

(The `Trades` section of your script defines these, so a customized export can
rename or add columns — treat the list above as the stock layout, not a fixed
schema.) Two things to know before mapping it:

- **P&L is in percent and dollars, not R.** `PctGain` is the return as a percent
  of position size and `Profit` is net dollars — neither is an R-multiple. There
  is **no per-trade risk column** in the export, so choosing the 1R denominator
  is a real modeling decision you own, not a column rename.
- **MFE/MAE *are* exported**, as `PctMFE` / `PctMAE` (percent runup / drawdown
  during the trade). Put them in the same R unit as the return.

If you define 1R as a fixed fractional risk `R_PCT` (e.g. you risked ~1% of the
position per trade), the conversion is one division each:

```python
import pandas as pd
from crucible.edge import TradeLog, edge_report, reality_check

raw = pd.read_csv("trades.csv")                 # RealTest SaveTradesAs output

R_PCT = 1.0                                      # your 1R, as a % move — YOUR choice
raw["r"]   = raw["PctGain"].str.rstrip("%").astype(float) / R_PCT
raw["mfe"] = raw["PctMFE"].str.rstrip("%").astype(float)  / R_PCT
raw["mae"] = raw["PctMAE"].str.rstrip("%").astype(float)  / R_PCT

trades = TradeLog.from_frame(
    raw, mapping={"DateIn": "entry_date", "DateOut": "exit_date", "Bars": "bars_held"},
)
print(edge_report(trades))
print(reality_check(trades))                     # the verdict RealTest's MC doesn't give
```

Map your columns onto the [`TradeLog`](src/crucible/edge/trade_log.py) schema
(`r` required; `mfe`, `mae`, `bars_held`, `prob`, `entry_date`, `exit_date`
optional) with `mapping=`. Two honesty notes that matter more than the plumbing:
without a defined per-trade risk, the R denominator is an assumption the whole
report rides on — pick it deliberately. And if the RealTest run is a
multi-strategy portfolio where several sub-strategies take the same trade, those
rows are duplicated and correlated; crucible's bootstrap assumes independent
trades, so de-duplicate (or collapse to one book) first or the confidence
interval will look tighter than reality.

*(Comparison verified against RealTest's published documentation as of
July 2026. RealTest is actively developed; check its docs for the current
feature set.)*

## Ecosystem

crucible is one piece of a small, deliberately unbundled toolchain. Each part
owns one layer and stops there:

- **[cotdata](https://github.com/mspinola/cotdata)** — the *data* layer. A local,
  file-based store for futures prices and CFTC Commitments-of-Traders positioning,
  with a producer/consumer split so research, backtests, and dashboards all read
  identical data. Point it at a store, `import cotdata`, and you have the OHLC
  frames and COT series you build a signal from.
- **crucible** *(this package)* — the *edge* layer. Turn that signal into a trade
  log and find out whether the edge is real.

The flow runs one direction: **`cotdata` (data) → your signal → `crucible`
(edge)**. Neither imports the other, so either works alone — but together they
take you from raw CFTC/price data to a capital-free verdict with no vendor SDK at
runtime.

## Releasing

Releases publish to PyPI via GitHub Actions using **Trusted Publishing** (OIDC —
no API tokens are stored anywhere). Changes are tracked in
[`CHANGELOG.md`](CHANGELOG.md).

**One-time setup** (maintainer, before the first publish):

1. Create two GitHub environments — repo **Settings → Environments** — named
   `pypi` and `testpypi`. (Add a required-reviewer rule on `pypi` for a manual
   approval gate, if you want one.)
2. Register a **pending Trusted Publisher** at
   <https://pypi.org/manage/account/publishing/>:
   PyPI project `crucible-quant`, owner `mspinola`, repo `crucible`, workflow
   `release.yml`, environment `pypi`. Repeat on
   <https://test.pypi.org/manage/account/publishing/> with environment
   `testpypi` for dry runs.

**Cutting a release:**

1. Bump `version` in `pyproject.toml` and move the `CHANGELOG.md` entry from
   *Unreleased* to the new version.
2. (Optional dry run) **Actions → Release → Run workflow → `testpypi`**.
3. Tag and push — the tag **must** match the `pyproject` version or the run fails:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0     # builds, twine-checks, publishes to PyPI
   ```

## License

MIT
