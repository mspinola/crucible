# The Edge Monitor: has a validated edge decayed?

!!! note "Status: spiked. `crucible.validation.monitor` exists and is tested."
    The module, its five `Thresholds` entries, and 31 tests are in the tree. What is
    **not** settled is the seam question in [Open questions](#open-questions-for-a-maintainer):
    whether a monitor belongs in crucible at all. Treat the code as an argument made
    concrete, not as a decision already taken.

    Two claims in the first draft of this page were wrong and are corrected below,
    both found by measuring rather than reasoning: the Gaussian ARL approximation
    survives skewed trade returns far better than predicted, and the reference
    implementation's detection latency is a **median** against a **mean**.

The gauntlet answers "is this edge real?" once, over a fixed log. It has nothing to
say about the question that follows a promotion: **is it still real?** That question
currently has no home in crucible, and only a partial one anywhere in the stack.

## The provocation

A practitioner's "Edge Monitor" over a live COT book, reduced to its structure:

```
CRS:  IS n=2603  OOS n=1321
  trades/yr:  IS 153  OOS 138        (opportunity set)
  expectancy (% of acct/trade):  IS +0.134%  OOS +0.158%   diff t-stat +0.8
  win rate:  IS 43%  OOS 42%
  rolling-200-trade expectancy at year-ends (alarm if < +0.067%):
    2017: +0.312% ... 2020: +0.028% <-- BELOW THRESHOLD ... 2026: +0.153%
  CUSUM calibrated: k=+0.101  h=29.7
    (false alarm 20%/decade; detects a true halving in median 474 trades = 37 months)
  CUSUM alarms in OOS: 0    current level: 0% of threshold
  POLICY VERDICT: FULL SIZE  [rolling now +0.153% vs threshold +0.067%]
```

The governing idea: freeze the in-sample expectancy at promotion, track the recent
expectancy against it, and cut size by half if the recent number falls to half the
baseline.

The design is more careful than most. The reference value is the textbook one:
`k = 0.101` against a baseline of `0.134` is `0.75 x mu_0`, which is exactly the
midpoint `(mu_0 + mu_1)/2` for a target shift of `mu_1 = mu_0/2`. Someone did the
CUSUM properly rather than picking a round number. The defects below are about what
the monitor is anchored to and what it is blind to, not about its arithmetic.

## What crucible already computes

Roughly half of that output is `crucible.validation.holdout` under a different name.

| Line in the reference output | Where it lives today |
|---|---|
| `IS n / OOS n`, IS vs OOS expectancy, difference test | [`holdout`](https://github.com/mspinola/crucible/blob/main/src/crucible/validation/holdout.py). Strictly better: a bootstrap CI and p-value per side rather than a t-stat, which matters because per-trade R is not normal and a t-statistic on it is optimistic in the tails |
| win rate, expectancy, per-trade dispersion | `edge.metrics`, `edge.stats.reality_check` |
| `trades/yr IS vs OOS` | nowhere, though `TradeLog` carries `entry_date` and could |
| rolling-200-trade expectancy series | nowhere. `windowed_segments` is the nearest relative, but it buckets by calendar era, not by a rolling trade count |
| CUSUM (k, h, ARL design) | nowhere in crucible, and nowhere in the wider stack |
| `POLICY VERDICT: FULL SIZE` | correctly absent. Sizing is a capital decision and belongs downstream |

So the genuinely new machinery is small: a rolling-window expectancy series, a
calibrated sequential detector, and a frequency channel.

## Why this is not `orchestrate.drift`

`crucible_stack.orchestrate.drift` already monitors a live book against a frozen
block-bootstrap envelope, and it is easy to read this proposal as a duplicate. It is
not. The two watch different things and come apart in both directions.

| | `orchestrate.drift` | Edge Monitor |
|---|---|---|
| Watches | the equity **path**: cumulative R and running drawdown | the per-trade **parameter**: expectancy, and firing rate |
| Fires when | the realized path leaves the p5 band at the current elapsed period | the recent per-trade edge has shifted down from its frozen baseline |
| Misses | expectancy halving while the path goes flat inside a wide band | a fat-tail drawdown cluster with expectancy fully intact |

A book whose expectancy halves but whose trade count is unchanged produces a path that
drifts flat. Against a p5 band provisioned over a multi-year horizon, flat is usually
still inside. Conversely a run of correlated losers can breach the drawdown floor
while every per-trade statistic is where it should be. Path monitoring and parameter
monitoring are complements.

## Five defects in the reference design

These are the things worth fixing rather than copying.

### 1. The baseline is the optimized number

The author is explicit that in-sample is "where all the parameters were optimized, so
it's more or less perfect." Anchoring the tripwire to an optimization-inflated
expectancy means "50% of baseline" does not mean 50% of the edge you actually have.

This is the single most crucible-shaped contribution available here. The package
already produces the corrected version of exactly that figure: `deflated_sharpe`,
`sidak_correction`, and the honest N from a `SearchSpaceLog`. A monitor whose baseline
is a **deflated** expectancy is measuring decay from a number that was defensible in
the first place.

Worth flagging in the reference numbers: OOS expectancy (`+0.158%`) is *above* IS
(`+0.134%`). That is backwards from the usual optimization bias. Either the search was
narrow (small honest N), the in-sample window was hostile, or the out-of-sample period
was favorable. A `SearchSpaceLog` count is what distinguishes those three, and without
one the reading is unresolvable.

### 2. Two alarms, one of them uncalibrated

The CUSUM has a stated false-alarm rate (20% per decade). The "50% of baseline" rolling
rule has none. In 2020 the two disagreed: the rolling rule crossed, the calibrated
detector recorded zero alarms across the whole out-of-sample period. The `POLICY
VERDICT` line reads off the uncalibrated rule.

Running both is defensible (one fast and noisy, one slow and calibrated). Running both
without declaring in advance which one governs the size decision is not.

### 3. The year-end reads are not independent looks

At 138 trades per year a 200-trade window spans about 17 months. Consecutive year-end
reads therefore share `200 - 138 = 62` trades, about 31% of the window. "Only one dip
in ten years" understates how often the rule fires on a perfectly stable edge, because
those ten reads carry far less than ten reads' worth of independent information.

### 4. Expectancy alone is blind to a frequency collapse

The reference output prints `trades/yr: IS 153 OOS 138` and then does not use it. A
signal that quietly stops firing halves annual R with per-trade expectancy untouched,
and this monitor prints FULL SIZE throughout.

It did not happen here. Annual throughput actually rose slightly:

```
IS   153 x 0.134% = 20.5% of account per year
OOS  138 x 0.158% = 21.8% of account per year
```

But the monitor would not have caught it if it had. Opportunity-set decay is a distinct
failure mode from edge decay and needs its own channel.

### 5. The units are capital-denominated

Everything is in "% of account per trade." Change the risk-per-trade fraction and the
entire series shifts for reasons that have nothing to do with the edge. In R the same
monitor is invariant to that, which is the whole reason `TradeLog` is denominated in R
and the reason crucible can judge without knowing account size.

## Proposed design

### The seam

Split it the way `orchestrate/drift.py` already splits itself. That module's header
reserves its R-space core as "designed to migrate into crucible if it earns its way,"
which is precisely the shape being proposed here.

| Concern | Home | Why |
|---|---|---|
| the statistic and the verdict | **crucible** | a `TradeLog` plus a frozen baseline in, a label out. No clock, no state, no side effects, seeded and deterministic |
| freezing the baseline at promotion | `orchestrate` / `livebook` | that is a moment in time and a durable record |
| the size decision | `crucible_stack.capital` / `orchestrate` | capital-aware by definition |

crucible emits HOLDING / SLIPPING / DEGRADED. It does not emit "cut to half size."

### What shipped

```python
from crucible.validation import EdgeBaseline, cusum_design, edge_monitor, empirical_arl

# ONCE, at promotion. Freeze the result.
base = EdgeBaseline.from_log(validated_log, deflated_expectancy=0.08, n_variants=64)

design = cusum_design(base)          # k and h derived from Thresholds, not typed in
verdict = edge_monitor(live_log, base)
print(verdict)                       # HOLDING | SLIPPING | DEGRADED
```

`rolling_expectancy(trades, window)` is the descriptive series, deliberately separate
from the detector that renders the verdict. `empirical_arl` resamples your own returns
to check the design's Gaussian claims (below).

All five knobs live in `Thresholds` (`monitor_detect_shift`, `monitor_arl0_trades`,
`monitor_window`, `monitor_slip_ratio`, `monitor_min_frequency_ratio`), never inline.

The design reproduces the reference implementation's reference value exactly: for a
baseline of `0.134` and a target halving, `k = 0.75 x mu_0 = 0.1005`, against a published
`k = 0.101`. That is a useful cross-check on an independent implementation, and it is a
test (`test_reference_value_is_the_textbook_midpoint`).

### Two things the first draft of this page got wrong

**The Gaussian assumption is much less of a problem than claimed.** The draft said the
nominal false-alarm rate would be "materially wrong" on fat-tailed returns. Measured, it
is not. Across an ordinary 43%-win-rate shape and a lottery-shaped book (10% win rate,
+12R winners), empirical ARL0 stays within 0.96x to 1.17x of nominal, and the drift is
in the conservative direction. The reason is structural: the boundary sits 7 to 30 sigma
out, so the CUSUM aggregates hundreds of increments before it can alarm and the central
limit theorem carries it. The approximation is weakest where `h` is small, which is
where `empirical_arl` earns its place. This is now a measured bound in the module
docstring rather than a hedge.

**ARLs are means, and the reference implementation quotes a median.** Its "median 474
trades = 37 months" cannot be reconciled with its stated `h = 29.7` under any single
sigma if read as a mean; as a median it hangs together, because the run-length
distribution is strongly right-skewed. Measured here, the median runs about a third
below the mean (7,520 mean against 4,980 median on one in-control design). Quoting one
against the other misstates detection latency badly, so `empirical_arl` returns both and
`CusumDesign.arl0` / `.arl1` are documented as means.

### The traps, and how each is held shut

Three, each with a test that fails if the guard is removed.

**Re-baselining.** A baseline recomputed from current data at comparison time re-fits
onto the drifted reality and the monitor can never fire. It looks entirely correct in
review and passes any test that does not span a real decay event. So `edge_monitor`
takes an `EdgeBaseline` and has **no parameter that could rebuild one**, asserted
directly on the signature by `test_edge_monitor_cannot_rebuild_a_baseline`.

**A silently undeflated baseline.** Defect 1 is invisible at the call site: passing the
raw in-sample expectancy produces a monitor that runs, prints, and is wrong about how
much room it has. `EdgeBaseline.deflated` therefore rides in the verdict output rather
than being validated away, on the same principle as `variant_count()` refusing a
typed-in int. An undeflated monitor is allowed. An undeflated monitor that does not say
so is not.

**An uncalibrated rule governing the decision.** Only the CUSUM can return `DEGRADED`.
The rolling ratio and the firing-rate ratio cap out at `SLIPPING`. This is the part
worth keeping if nothing else here survives review, and the demo shows why: on a book
whose true edge was fully intact and slightly **above** baseline, a 200-trade trailing
read still printed "59% of baseline" from noise alone. Had that rule governed sizing,
it would have cut a healthy book.

## Open questions for a maintainer

1. **Does this belong in crucible at all?** In the chain of refusals, "is it still
   real?" is `orchestrate`'s question, not crucible's, and crucible has so far been a
   one-shot judge rather than a monitor. The case for crucible: a stateless CUSUM over
   a `TradeLog` owns no clock and persists nothing, so it satisfies every invariant the
   package enforces. This is a seam call and should be made deliberately, not inherited
   from wherever the code happened to get written.
2. **Label vocabulary.** `reality_check` already uses HELD / FRAGILE / FAIL. Reusing
   HELD here would collide on meaning. HOLDING / SLIPPING / DEGRADED is proposed to
   keep them distinguishable in a report that shows both.
3. **Is this a fifth gauntlet gate?** Probably not. The gauntlet gates a promotion
   decision from a fixed log, and this runs continuously afterwards. It is more likely
   a peer of the gauntlet than a member of it.
4. **What produces `sigma`?** The per-trade dispersion the CUSUM design needs is
   available from the same log the baseline came from, but whether it should be frozen
   alongside the baseline or re-estimated live is a real choice, and re-estimating it
   is a soft form of trap 1. Currently frozen.
5. **The firing-rate channel is uncalibrated.** It compares a ratio to a threshold, so
   it can only ever say `SLIPPING`, which is consistent but weak. A proper arrival-process
   test (trade arrivals are approximately Poisson) would give it a stated false-alarm
   rate and let it stand beside the CUSUM. Unbuilt.
6. **Nothing deflates an expectancy yet.** `deflated_expectancy` is a number you pass
   in. `deflated_sharpe` corrects a Sharpe ratio, not a per-trade mean, so the
   conversion is left to the caller. That is the largest remaining gap between this
   module and the argument that motivates it.

## Bottom line

The idea is sound and worth building, but it is not a new pillar of the gauntlet. The
in-sample versus out-of-sample comparison at the top of the reference output is
something crucible already does more honestly. What is genuinely missing is a rolling
expectancy series, a calibrated sequential alarm, and a frequency channel, which is a
modest amount of code.

The improvement crucible can offer over the reference version is not the detector. It
is what the detector is anchored to: a search-corrected expectancy instead of the
optimized in-sample one, denominated in R instead of percent of account, watched
alongside the opportunity set rather than in isolation. The 2020 dip in the reference
output is the uncalibrated alarm firing while the calibrated one stayed silent, which
is not evidence that the monitor works.
