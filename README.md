# crucible

[![PyPI](https://img.shields.io/pypi/v/crucible-quant)](https://pypi.org/project/crucible-quant/)
[![CI](https://github.com/mspinola/crucible/actions/workflows/ci.yml/badge.svg)](https://github.com/mspinola/crucible/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

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

> Runnable versions live in [`examples/`](examples): `quickstart.py` and
> `validation.py` use synthetic data (no network); `real_data_yfinance.py` pulls
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
variant you searched) for when a grid search flatters the best result.
See [`examples/validation.py`](examples/validation.py).

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
(`realtest -test script.rts`) and, with `SaveTradesAs` / `SaveStatsAs` set, writes
the trade log and summary stats to CSV. That CSV is exactly what crucible ingests:

```python
import pandas as pd
from crucible.edge import TradeLog, edge_report, reality_check

raw = pd.read_csv("trades.csv")                 # RealTest SaveTradesAs output

# RealTest reports P&L in dollars/percent; crucible's `r` is risk-normalized
# (1R = the risk taken at entry). Normalize by per-trade initial risk first —
# this is the one real step, not a plain column rename:
raw["r"] = raw["ProfitDlr"] / raw["RiskDlr"]    # adjust names to your columns

trades = TradeLog.from_frame(raw, mapping={"MFE": "mfe", "MAE": "mae"})
print(edge_report(trades))
print(reality_check(trades))                     # the verdict RealTest's MC doesn't give
```

The one thing to get right is the R normalization: RealTest's trade CSV is in
dollars/percent, so you divide each trade's P&L (and its MFE/MAE) by the initial
risk that trade took to land in R-multiples. A strategy that trades a fixed
stop distance makes this a single division; without a defined per-trade risk,
decide what your 1R denominator is before trusting the metrics. Column names in
the export vary by RealTest version and your results-script config — map them to
the [`TradeLog`](src/crucible/edge/trade_log.py) schema
(`r`, and optionally `mfe`, `mae`, `bars_held`, `prob`, `entry_date`, `exit_date`)
with `mapping=`.

*(Comparison verified against RealTest's published documentation as of
July 2026. RealTest is actively developed; check its docs for the current
feature set.)*

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
