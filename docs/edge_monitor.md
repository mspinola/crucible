# The Edge Monitor: has a validated edge decayed?

!!! note "Status: shipped. `crucible.validation.monitor`, merged in #109."
    The module, its five `Thresholds` entries, and 31 tests are on `main`. This page
    began as a pre-build proposal and is now a living description of what exists; the
    reasoning is kept because the *why* outlived the decision, but read
    [Still open](#still-open) rather than this page's history for what is missing.

    Two claims in the first draft were wrong and are corrected below, both caught by
    measuring rather than reasoning: the Gaussian ARL approximation survives skewed
    trade returns far better than predicted, and the reference implementation's
    detection latency is a **median** quoted against a **mean**. Those corrections are
    left in place deliberately. A design note that quietly edits out its own errors is
    less useful than one that records them.

The gauntlet answers "is this edge real?" once, over a fixed log. It has nothing to
say about the question that follows a promotion: **is it still real?** Until #109 that
question had no home in crucible, and only a partial one anywhere in the stack.

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

## Where each line of that output lives

Roughly half of it was already `crucible.validation.holdout` under a different name,
which is why the module that shipped is small.

| Line in the reference output | Where it lives | Pre-#109? |
|---|---|---|
| `IS n / OOS n`, IS vs OOS expectancy, difference test | [`holdout`](https://github.com/mspinola/crucible/blob/main/src/crucible/validation/holdout.py). Strictly better: a bootstrap CI and p-value per side rather than a t-stat, which matters because per-trade R is not normal and a t-statistic on it is optimistic in the tails | yes |
| win rate, expectancy, per-trade dispersion | `edge.metrics`, `edge.stats.reality_check` | yes |
| `trades/yr IS vs OOS` | `EdgeBaseline.trades_per_year`, derived from `entry_date`, and the monitor's third channel | no |
| rolling-200-trade expectancy series | `rolling_expectancy`. (`windowed_segments` is the nearest older relative, but it buckets by calendar era rather than by a rolling trade count) | no |
| CUSUM (k, h, ARL design) | `cusum_design`, checked against your own returns by `empirical_arl` | no |
| `POLICY VERDICT: FULL SIZE` | still correctly absent. Sizing is a capital decision and belongs downstream | n/a |

The genuinely new machinery was therefore small: a rolling-window expectancy series, a
calibrated sequential detector, and a frequency channel.

## Why this is not `orchestrate.drift`

`crucible_stack.orchestrate.drift` already monitors a live book against a frozen
block-bootstrap envelope, and it is easy to read this module as a duplicate. It is not.
The two watch different things and come apart in both directions.

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

`validation.deflated_expectancy` now does that conversion; see
[Deflating the baseline](#deflating-the-baseline) below.

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

## The design

### The seam

Split the way `orchestrate/drift.py` already splits itself. That module's header
reserves its R-space core as "designed to migrate into crucible if it earns its way,"
which is precisely the shape taken here.

| Concern | Home | Why |
|---|---|---|
| the statistic and the verdict | **crucible** | a `TradeLog` plus a frozen baseline in, a label out. No clock, no state, no side effects, seeded and deterministic |
| freezing the baseline at promotion | `orchestrate` / `livebook` | that is a moment in time and a durable record |
| the size decision | `crucible_stack.capital` / `orchestrate` | capital-aware by definition |

crucible emits HOLDING / SLIPPING / DEGRADED. It does not emit "cut to half size."

### What shipped

```python
from crucible.validation import (
    EdgeBaseline, cusum_design, deflated_expectancy, edge_monitor, empirical_arl,
)

# ONCE, at promotion. Freeze the result.
corrected = deflated_expectancy(validated_log.r, [t.r for t in trials], n_trials=64)
base = EdgeBaseline.from_log(validated_log, deflated_expectancy=corrected, n_variants=64)

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

**The Gaussian assumption, corrected twice.** The first draft of this page said the
nominal false-alarm rate would be "materially wrong" on fat-tailed returns. Measuring
synthetic books contradicted that, so the page was changed to claim it "stays within 0.96x
to 1.17x of nominal". Then the monitor met a real book, and that second claim was wrong
too. It generalized from a synthetic 10%-win-rate case with skew **+3**; real pooled
trend-following runs about **+5**, with single trades near +39R against losses capped
near -1R.

The error is a function of skew, and grows with the boundary:

| skew | empirical ARL0 vs nominal |
|---|---|
| +3 (the synthetic case the old claim rested on) | ~1.1x |
| +4 | ~1.5x to 1.7x |
| **+5 (a real pooled trend book)** | **~2.0x** |
| +8 | ~2.2x to 2.6x |
| +11 | ~2.9x to 4.0x |

The higher figure in each range is the larger `monitor_arl0_trades`, so a stricter
false-alarm budget is also where the stated number is least trustworthy.

The drift is conservative and, measured, free. Empirical ARL0 above nominal means fewer
false alarms than advertised, and the inflation does **not** carry over to `arl1`:
measured detection latency tracked nominal within a few percent at every budget tested on
the real book. The asymmetry is structural. In control the statistic hovers near zero and
alarms only via a rare large excursion, exactly where a fat right tail bites; under a real
shift it reaches the boundary by drift, where tail shape barely matters.

A draft of this section claimed the opposite, that `arl1` inflates too. That was reasoning
by analogy from the ARL0 result rather than measuring, and measuring contradicted it. It
is the third correction this page has recorded on the same paragraph, which is a fair
indication of how poorly this particular thing yields to intuition.

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
worth keeping if nothing else here survives review, and
[`examples/edge_monitor.py`](https://github.com/mspinola/crucible/blob/main/examples/edge_monitor.py)
shows why. On a book whose true edge is fully intact and **above** baseline,
the 200-trade trailing read ranges from -25% to 197% of baseline on noise alone and dips
under the 50% line in 11% of windows, while the CUSUM peaks at 82% of its threshold and
never fires. A "cut at 50% of baseline" rule would have cut a healthy book on whichever
window you happened to read.

## Deflating the baseline

Defect 1 was the largest gap on this page for as long as it stood: the argument for
building the monitor here rather than copying the reference implementation rested on
anchoring to a search-corrected number, and nothing in the package produced one.
`deflated_sharpe` corrects a Sharpe and returns a **probability**, which is the right
output for a gate and useless to a monitor. A monitor needs a number in R.

`validation.deflated_expectancy` writes the conversion, on the bar `deflated_sharpe`
already uses:

```
SR0      = expected MAXIMUM per-trade Sharpe of N noise trials
           (Bailey/López de Prado, scaled by the spread of the trial Sharpes)
deflated = mu - sigma * SR0
```

The bar lives in Sharpe units, so it is carried back into R by the winner's own sigma
before being subtracted. Both functions now call one `_expected_max_sharpe`, so the two
corrections for one search cannot disagree about how big the search was.

```python
from crucible.validation import deflated_expectancy, EdgeBaseline

d = deflated_expectancy(winner.r, [t.r for t in every_variant_tried], n_trials=log)
base = EdgeBaseline.from_log(winner, deflated_expectancy=d, n_variants=log.n_variants)
```

Three decisions worth recording.

**It takes trial LOGS, not trial Sharpes**, unlike `deflated_sharpe`. The Sharpes are
computed inside, so their clock cannot be got wrong. A per-month Sharpe and a per-trade
Sharpe are different numbers on different scales, and multiplying the wrong one by a
per-trade sigma produces a haircut in no units at all, silently. That is the v0.4.0 units
bug in a new costume, and the fix is to not accept the ambiguous input.

**It is a bias correction, not a significance test, and the docstring says so in those
words.** It removes the selection bias a search of this size is *expected* to produce.
The realized maximum sits above its own mean about half the time, so a pure-noise winner
still clears zero here roughly as often as not: measured at 56% / 47% / 44% for N = 5 /
20 / 100, while `deflated_sharpe` correctly calls 0% of the same draws significant
(reproducer:
`tests/test_deflated_expectancy.py::test_the_haircut_is_a_bias_correction_not_a_test`).
The result object's property is therefore named `is_positive` rather than `survives`,
because the first draft called it `survives` and that reads as a verdict it does not
deliver. Establish the edge is real with the gauntlet; use this to decide what to anchor
to afterwards.

**It over-corrects a genuine edge, deliberately.** A winner chosen partly for real signal
carries less selection bias than the pure-luck maximum being subtracted, so the deflated
number sits below the truth. For a monitor baseline that is the safer direction: too low
a bar makes the monitor slow to call decay, too high a bar makes it cry wolf, and a
spurious alarm forces a re-optimization that taxes the honest N of the next verdict.

A correction that leaves nothing raises rather than returning a smaller baseline.
`EdgeBaseline` already refused a non-positive expectancy; the message now names deflation
as a cause, because a book whose edge does not survive its own search is not a monitoring
problem.

## Settled

These were the open questions this page carried before #109. Merging answered them, so
they are recorded here with what decided them rather than left looking live.

| Question | Decision | What decided it |
|---|---|---|
| Does a monitor belong in crucible at all? "Is it still real?" is nominally `orchestrate`'s question. | **Yes.** | Merging #109. The case that carried it: a stateless CUSUM over a `TradeLog` owns no clock and persists nothing, so it satisfies every invariant the package enforces, and `orchestrate/drift.py` already reserved its R-space core for exactly this migration. |
| Label vocabulary, given `reality_check` already uses HELD / FRAGILE / FAIL. | **HOLDING / SLIPPING / DEGRADED.** | Kept distinct so a report showing both verdicts cannot blur them. Reusing HELD would have collided on meaning. |
| Is this a fifth gauntlet gate? | **No.** | Nothing is wired into `run_gauntlet`. The gauntlet judges a promotion decision from a fixed log; this runs continuously afterwards, so it is a peer rather than a member. |
| Frozen or live `sigma`? | **Frozen**, alongside the baseline. | Re-estimating per-trade dispersion from live data is a softer form of the re-baselining trap: it lets the reference drift toward whatever is happening now. |
| Can the firing-rate channel be calibrated, so it may escalate to `DEGRADED` too? | **No, and it was tried.** | See [Why the firing-rate channel stays uncalibrated](#why-the-firing-rate-channel-stays-uncalibrated). Three detector families were built and measured against a real book's arrivals. All three deliver a false-alarm rate 2-3x worse than stated, and calibrating empirically leaves the delivered budget uncertain by **14.6x**. The channel stays capped at `SLIPPING`. |

## Why the firing-rate channel stays uncalibrated

This page carried "an arrival-process test would give it a stated false-alarm rate" at
the top of [Still open](#still-open) for as long as it existed, on the reasoning that
trade arrivals are approximately Poisson. **That reasoning was wrong, and the claim is
removed rather than softened.** What follows is what measuring it actually produced, so
nobody builds it twice.

The prize was real. On a live 47-market trend book firing 33.9 trades/yr, a count-based
detector spots a **halved firing rate in 0.6 years**, against 5.1 years for the
expectancy CUSUM at the same budget. Roughly eight times faster, and it covers the one
failure the other two channels cannot see. It is worth wanting.

**Arrivals are not Poisson.** Dispersion index 3.43 against Poisson's 1.0, on annual
counts over the full log. About 40% of that is a secular trend (+0.49 trades/yr per
year, r=+0.63); the rest is genuine sub-annual clustering, with lag-1 autocorrelation of
detrended annual counts at +0.04, so the years themselves are independent.

Three detectors were built and checked against the book's own arrivals, each designed
for a **10-year** false-alarm budget:

| Detector | Delivered ARL0 | vs stated |
|---|---|---|
| Exponential CUSUM on inter-arrival gaps, full-span window | 3.6 yr | 0.36x |
| ...same, 7-year window | 3.5 yr | 0.35x |
| ...same, 5-year window | 1.8 yr | **0.18x** |
| Poisson CUSUM on monthly counts | 3.2 yr | 0.32x |

Every one fires 3x to 5x more often than advertised, and in the **dangerous** direction.
Compare the expectancy CUSUM, where skew inflates ARL0 by 1.39x and buys free margin;
here the error spends margin instead.

Two intermediate hypotheses were tested and both failed, which is why the table above
has more rows than the argument strictly needs:

- **"A recent window will restore the Poisson model."** Annual-count dispersion does fall
  sharply in recent windows (0.68 at 7 years, against 3.43 pooled). But a CUSUM runs on
  individual gaps, not annual counts, and those stay non-exponential at every window. The
  coarse-scale and fine-scale properties come apart, and the 5-year window is the worst of
  the lot.
- **"Then calibrate `h` empirically instead of analytically."** This is the honest fallback
  and it is what `empirical_arl` already does for R. Solving `h` against the book's real
  monthly counts gives 6.22 where the analytic design gives 4.26, a 1.46x correction. But
  bootstrapping that calibration, `h` ranges 4.15 to 8.18 across replicates, and holding
  `h` fixed the **delivered budget ranges 5.3 to 77.9 years**. A stated 10-year rate that
  is truly somewhere in 5-78 is not a stated rate.

The root cause is not the model, it is the data. A book firing 33.9 trades/yr gives 82
monthly observations in a 7-year window, and CUSUM run length depends on the tail of the
count distribution, which 82 samples cannot pin down. The gap detector is worse still:
crossing its boundary requires two or three of the largest gaps back to back, so it is a
rare-combination test over the tail of a 233-gap sample rather than a drift detector, and
its ARL curve is visibly stepped as a result.

**So the channel keeps its ratio rule and its `SLIPPING` cap.** An uncalibrated rule that
says it is uncalibrated is more honest than a calibrated-looking one that is wrong by 3x,
and the rule this page most wants to keep is that only a detector with a stated
false-alarm rate may escalate. Manufacturing a stated rate to satisfy that rule would
defeat it.

What would change the answer is more arrivals, not better statistics: a book firing
several hundred trades a year would have enough periods to calibrate. That is a property
of the book, so the honest place to revisit this is a higher-frequency one, not a cleverer
detector here. `tests/test_frequency_calibration.py` pins the general claim on synthetic
data, so the limit can be re-derived without the private book.


## Still open

Ordered by how much each one undercuts the argument for the module.

1. **It has never met real decay.** Only synthetic decay, generated to test it. The ARL
   figures are design targets, not field results. The first honest test is the first
   promoted book that genuinely degrades.
2. **The README still does not mention the module,** though it enumerates every other
   subpackage's API with a worked block. The worked example, tutorial §14 and the
   `report.monitor_panel` block all exist now; the README is the last documentation gap.

## Bottom line

This is not a new pillar of the gauntlet, and it was never meant to be. The in-sample
versus out-of-sample comparison at the top of the reference output was something
crucible already did more honestly; what was missing was a rolling expectancy series, a
calibrated sequential alarm, and a frequency channel, which turned out to be a modest
amount of code.

The improvement over the reference version is not the detector. It is what the detector
is anchored to: a search-corrected expectancy instead of the optimized in-sample one,
denominated in R instead of percent of account, watched alongside the opportunity set
rather than in isolation. All three are now built, the first of them last and only after
this page had carried it as the top open item for several revisions.

The 2020 dip in the reference output is the uncalibrated alarm firing while the
calibrated one stayed silent. That is not evidence the monitor works, and the same
pattern reproduces here: in §14 of the tutorial, a book whose edge never decayed shows a
trailing read swinging between -25% and 197% of baseline. Hence the rule that only a
detector with a stated false-alarm rate may escalate to `DEGRADED`.
