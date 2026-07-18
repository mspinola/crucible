# Run modes: from a quick read to the full gauntlet

crucible judges the same trade log at **several levels of strictness**. The cheapest
mode reads the whole history and asks only *"is there an edge, or is this noise?"* The
strictest runs the full gauntlet and asks *"is it real, strong, durable, and general
enough to fund?"* Each mode is a higher bar. A genuine edge should clear all of them. A
fragile one drops out at the level that matches its flaw, and that *where* is the
diagnosis.

| Mode | The question it answers | What you call | Output |
|---|---|---|---|
| **Full range** | Is there an edge at all, or small-sample noise? | `edge_report` + `reality_check` | console report + a `tearsheet()` (HELD / FRAGILE / FAIL) |
| **Holdout** | Does it survive an early-train / late-confirm split? | `validation.holdout` | console scorecard (TRAIN vs the untouched TEST) |
| **Gauntlet** | Real, strong, durable, general: the deployable verdict? | `validation.run_gauntlet` | console audit + a `gauntlet_report()` tearsheet |

Below, one strategy runs through all three. It is a **Donchian channel breakout** that
looks great and does not survive: go long when price closes above the prior 20-bar high,
exit on a 2.5R target, a 1R stop, or a 30-bar cap. The full runnable script is
[`examples/donchian_gauntlet.py`](https://github.com/mspinola/crucible/blob/main/examples/donchian_gauntlet.py),
on reproducible synthetic prices, so these are the exact numbers you get.

```python
from crucible.edge import barrier_trades, edge_report, reality_check
from crucible.validation import holdout, walk_forward, run_gauntlet

def donchian(df, lookback=20):
    return df["Close"] > df["High"].rolling(lookback).max().shift(1)

entries = donchian(px, lookback=20)                                  # your signal
trades  = barrier_trades(px, entries, side="long", tp=2.5, sl=1.0, timeout=30)
```

## Full range: the whole-history read

The cheapest mode. Describe the edge, then ask the bootstrap whether it clears zero,
over the entire trade log at once.

```python
edge_report(trades)      # the capital-free scorecard
reality_check(trades)    # is the expectancy distinguishable from zero?
```

```
edge_report(trades)
   Trades        : 162          Payoff ratio  : 2.50
   Win rate      : 43.2 %       SQN-100       : 2.95
   Expectancy    : +0.512 R     Excursion     : 1.77   [PASS]
   Profit factor : 1.90         Time asymmetry: 2.19   [PASS]

reality_check(trades)
   VERDICT (expectancy): +0.512 R   95% CI [+0.253, +0.772]
                         p(edge>0) = 1.000   ->  HELD
```

The CI lower bound (**+0.253**) clears zero, and across 10,000 resamples the edge stays
positive. Verdict **HELD**: on the whole history this is not small-sample noise. The
same read renders as a shareable tearsheet, the verdict banner, the metric row, and the
edge panels behind it:

![A crucible full-range tearsheet for the Donchian breakout: a green HELD banner (expectancy +0.512R, CI [+0.253, +0.772]), the metric row, the R-multiple distribution, a cumulative-R curve rising to about 80R, the MFE/MAE excursion scatter, and a bootstrap-expectancy histogram sitting entirely right of zero.](img/donchian_tearsheet.png){ width="640" }

A backtester would stop right here, with that rising curve, and sell you the system.
crucible does not: **HELD on the pooled log is necessary, not sufficient.** It says the
edge is real *somewhere in this history*, not that it holds up going forward.

## Holdout: does it survive out of sample, in time?

A stricter bar. Fit on an early slice, confirm on a later one the analysis never
touched, with a purge/embargo band so a straddling trade cannot leak across the split.

```python
holdout(trades, "2016-01-01", embargo_weeks=8)   # verdict = the untouched TEST half
```

```
HOLDOUT @ 2016-01-01 (embargo 8w)
  TRAIN  n=77   E=+0.545R  CI[+0.182, +0.955]  p(edge>0)=0.999  [HELD]
  TEST   n=84   E=+0.500R  CI[+0.125, +0.875]  p(edge>0)=0.997  [HELD]
  -> HELD  (verdict = the untouched TEST period)
```

The TRAIN half looks good, as it should, that is where an edge would have been chosen.
The honest read is the **TEST** half, and it is **HELD** too: expectancy +0.500R with a
CI clear of zero on 84 trades the split kept hidden. So a single clean early/late split
does *not* catch this breakout either. (Holdout is a console scorecard. crucible does not
render a tearsheet for it.)

## Gauntlet: the deployable verdict

The full bar. `run_gauntlet` runs the ordered gates and, crucially, adds **DURABLE**, a
walk-forward check the two modes above never applied. Walk-forward re-optimizes the
lookback on each in-sample window, applies the winner to the next unseen year, and
stitches the out-of-sample slices into one honest log.

```python
wf = walk_forward(px, donchian, {"lookback": [20, 40]}, is_days=3*365, oos_days=365)
run_gauntlet(wf.stitched, prices=px, wf=wf, n_variants=2)
```

```
REAL     ✓   corrected p = 0.0008   ·   beats 100% of random-entry timing
STRONG   ✓   expectancy CI-low +0.40 · PF CI-low 1.67 · SQN CI-low 2.32
DURABLE  ✗   wfe_aggregate 1.34 above the [0.30, 1.00] ceiling  ·  fold_dispersion CV 1.51 → PASS
─────────────────────────────────────────────────────────────────────
GAUNTLET: FAIL   (2 of 3 gates passed, failing: DURABLE)
```

![The gauntlet report for this run: a GAUNTLET FAIL banner with pillar chips (REAL and STRONG pass, DURABLE fails), the metric row, and a plain-English 'Not validated' verdict.](img/gauntlet_hero.png){ width="640" }

REAL and STRONG both pass: the breakout is not noise (sign-permutation p = 0.0008,
Šidák-adjusted for the two lookbacks tried) and clears every metric at its pessimistic CI
lower bound. But **DURABLE** kills it. The aggregate walk-forward efficiency is 1.34,
*above* the 1.00 "too-good-to-be-true" ceiling: the out-of-sample outran the in-sample,
inflated by a few outlier years, the opposite of the graceful degradation a robust edge
shows. The stitched out-of-sample curve is convincing, and the gauntlet rejects it anyway:

![Cumulative R of the stitched out-of-sample log: an early climb, a long flat and choppy middle, then a late rip up to about 78R.](img/gauntlet_cumr.png){ width="640" }

## The lesson

The same trade log, three modes, three different bars:

| Mode | Donchian result |
|---|---|
| **Full range** | **HELD** (+0.512R, CI clears zero) |
| **Holdout** | **HELD** (TEST +0.500R on the untouched half) |
| **Gauntlet** | **FAIL** (DURABLE, the walk-forward exposes the overfit) |

The cheap modes both pass. Only the gauntlet's walk-forward durability gate catches the
flaw, an edge that is real in aggregate but does not hold up *through time*. That is the
whole argument for the fuller mode: **run the bar that matches the claim you want to
make, and run the gauntlet before you fund anything.**

The gauntlet bundles more gates than shown here (GENERAL for cross-market reach, and
optional PBO / deflated-Sharpe checks on the search itself). The
[tutorial](tutorial.md) works each underlying technique end to end, and
[the gauntlet design page](edge_gate.md) documents every gate and its thresholds.
