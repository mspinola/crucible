---
title: Home
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![crucible](img/crucible_logo.png){ .hero-logo }

# Measure the edge *before* you ever open a $100k account.

Most trading "edges" are artifacts of a small sample. **crucible** takes a
**trade log** and tells you, with a confidence interval and a p-value, whether
the edge is real. No account, no position sizing, no equity curve. It's the
thing you run *before* a backtester.

[Get started](#quickstart){ .md-button .md-button--primary }
[Read the tutorial](tutorial.md){ .md-button }
[View on GitHub](https://github.com/mspinola/crucible){ .md-button }

[Run modes](run_modes.md){ .md-button }
[Visualizations](visualizations.md){ .md-button }
[The gauntlet (design)](edge_gate.md){ .md-button }
[Architecture](architecture.md){ .md-button }

</div>

## Quickstart

```bash
pip install crucible-quant              # core: metrics + stats + simulator (numpy/pandas only)
pip install "crucible-quant[examples]"  # + yfinance, to run the demo below on real data
```

> Installed as **`crucible-quant`**, imported as **`crucible`** (`import crucible`).

## The 30-second example

```python
import yfinance as yf
from crucible.edge import barrier_trades, edge_report, reality_check
from crucible.strategies import ma_cross

px = yf.download("ES=F", start="2010-01-01")        # any OHLC frame works
entries = ma_cross(px, fast=20, slow=50)            # your signal: a boolean Series

trades = barrier_trades(px, entries, side="long",   # signal -> trade log (in R)
                        tp=2.0, sl=1.0, timeout=20) # 2R target, 1R stop, 20-bar cap

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
  point positive, but the CI straddles zero, not distinguishable
  from noise at this sample size. Do NOT size it up.
```

That `FRAGILE` block is the whole point: a positive expectancy that a
backtester would have shown you as a rising equity curve is, at this sample
size, **indistinguishable from noise**. crucible says so out loud.

## What crucible answers

From a single trade log up to the whole correlated book, one question at a
time, each harder than the last, each answered out loud by a named function:

<div class="grid cards" markdown>

-   :material-clipboard-text-outline:{ .lg .middle } **Describe the edge**

    ---

    Expectancy, profit factor, payoff, SQN, MFE/MAE efficiency — the
    capital-free scorecard.

    `edge_report`

-   :material-chart-bell-curve:{ .lg .middle } **Quantify sampling noise**

    ---

    A bootstrap CI and a p-value on every headline metric, so a point estimate
    becomes a defensible claim.

    `bootstrap_ci` · `p_value_positive`

-   :material-dice-multiple-outline:{ .lg .middle } **Rule out data-mining luck**

    ---

    Sign-permutation p-value (Masters) plus a random-entry reality check:
    could no-edge-at-all have produced this?

    `permutation` · `reality_check`

-   :material-chart-timeline-variant:{ .lg .middle } **Rule out drift**

    ---

    A leakage-controlled early-train / late-confirm split, with purge and
    embargo hygiene at the boundary.

    `validation.holdout`

-   :material-repeat:{ .lg .middle } **Confirm out-of-sample**

    ---

    Pardo walk-forward: optimize in-sample, confirm out-of-sample, stitch the
    honest log.

    `validation.walk_forward`

-   :material-magnify-expand:{ .lg .middle } **Price the search itself**

    ---

    How much did *selecting* the best config overfit? Probability of Backtest
    Overfitting (CSCV) and the deflated Sharpe.

    `validation.pbo`

-   :material-vector-link:{ .lg .middle } **Account for correlation**

    ---

    The effective number of *independent* bets across a correlated book — the
    honest denominator for significance.

    `breadth.effective_n`

-   :material-gate:{ .lg .middle } **…then all of it, as one gate**

    ---

    Every check above as an ordered, audited gauntlet that passes only if
    every gate does.

    `validation.run_gauntlet`

</div>

## One verdict for the whole edge

The individual tools each answer one question. The **gauntlet** runs them as an
ordered set of hard gates — REAL (not noise, corrected for the search) →
STRONG (real at the CI lower bound) → DURABLE (holds out-of-sample) →
GENERAL (travels across markets) — and returns a single audited pass/fail.

```python
from crucible.validation import run_gauntlet

gauntlet = run_gauntlet(
    wf.stitched,        # the honest log, stitched out-of-sample
    prices=px,          # enables REAL's random-timing null
    wf=wf,              # adds the DURABLE gate
    n_variants=4,       # size of your search -> REAL's Šidák correction
)
print(gauntlet.audit_report())
print(gauntlet.passed)  # True only if every gate that ran passed
```

[How the gauntlet is designed →](edge_gate.md)

## What this is, and isn't

✅ Trade-level edge metrics, excursion efficiency, bootstrap CIs, a
random-entry reality check, all **capital-free**. Costs are in scope: test on
`r` **net of your commission and slippage** — the
[tutorial provides a simple per-trade cost model](tutorial.md#1-the-substrate-a-risk-normalized-trade-log-r-multiples)
for the netting, or supply your own.

❌ No capital, position sizing, CAGR, drawdown, or Monte-Carlo-on-equity. *If
you want an equity curve, hand the `TradeLog` to
[quantstats](https://github.com/ranaroussi/quantstats). crucible stops at the
edge.*

## Go deeper

📖 **[The tutorial](tutorial.md)** — *From Trade Log to Verdict: the
statistics of a significant edge*. Every technique worked end to end on a
Donchian breakout, with references to the source literature.
Prefer offline? [Download it as a PDF](https://mspinola.github.io/crucible/tutorial.pdf).

## Ecosystem

**The only input crucible needs is a trade log.** Build one with the bundled
barrier simulator, export it from your backtester, or map any frame onto
`TradeLog` — crucible hands back the verdict either way, whatever your data
source.

If you also need futures / COT data to build a signal from, there's an
optional companion:

- **[cotdata](https://github.com/mspinola/cotdata)** — an *optional* data
  layer: a local, file-based store that wraps Norgate, Databento, or yfinance
  behind one read API, extensible to other data providers.

The flow runs one direction: **`cotdata` (data) → your signal → `crucible`
(edge)**. Neither imports the other — crucible works alone with any source of
trades.

---

Open source under the [MIT license](https://github.com/mspinola/crucible/blob/main/LICENSE)
· [Changelog](https://github.com/mspinola/crucible/blob/main/CHANGELOG.md)
· [PyPI](https://pypi.org/project/crucible-quant/)
