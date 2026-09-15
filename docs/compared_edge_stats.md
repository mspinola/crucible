# Compared: LuxAlgo Edge Stats

!!! note "Status: comparison, and one design sketch"
    Written 2026-09-15 against the public page at
    [luxalgo.com/edge-stats](https://www.luxalgo.com/edge-stats/) and its repository
    ([LuxAlgo/edge-stats](https://github.com/LuxAlgo/edge-stats), MIT code, CC BY 4.0
    bundled data). Feature claims below are the page's own words at that date; the
    product will move and this page will not follow it. The second half sketches the one
    thing Edge Stats reports that crucible does not, time to outcome, and what it would
    look like here. Not built, by decision: most of what the block says is already in
    the tearsheet (`exit_reason_breakdown`, `holding_vs_r`), no current book asks the
    question, and anything added to `EdgeReport` joins the Pine conformance loop. The
    sketch is kept so the design is not redone when a study does ask.

Short version: the two answer different questions and sit at different points in the
pipeline. Edge Stats is a conditional-frequency lookup engine. crucible takes a trade
log and decides whether an edge survives correction for the search that found it. Edge
Stats is closer to something you would feed crucible than to a competitor.

## What Edge Stats is

- **Input.** Bars from seven adapters (Binance, Coinbase, Alpaca, Databento, Massive,
  CSV, a demo store), or your own fills imported through a "Broker SDK" ("every broker,
  one schema, your keys", read-only).
- **Query.** 42 presets over 11 categories (gap fill, opening range break, inside day,
  ...) plus a WHERE-style language:
  `gapFill WHERE dayOfWeek = Tue AND gapPct BETWEEN 0.2% AND 0.6%`.
- **Output.** Win rate with a Wilson 95% interval and N; a LOW SAMPLE flag under N=30
  and nothing at all under 10; minutes-to-outcome at the median, p25, p75 and p90; a
  per-year table; a "stability" line that compares the first and second half of the
  sample (`88.2% (n=51) vs 70.6% (n=51) · halves agree ✓`) and a 250-session recency
  view.
- **Delivery.** Everything is precomputed at sync time, so a multi-year query answers
  in about 20 ms. A local MCP server exposes ten read-only tools; three of them
  (`edge_symbols`, `edge_presets`, `edge_report`) are on a hosted, keyless MCP; there is
  a web dashboard on LuxAlgo's Vela chart engine.
- **Framing.** "Historical conditional frequencies with sample sizes, not predictions
  and not advice." No order execution, no personalised advice.

## Where the two overlap

Only at the descriptive layer. Win rate plus a Wilson interval plus N is
`crucible.edge.edge_report` with every column except win rate removed, and
"halves agree" is a weaker cousin of `holdout`. Both attach a sample size to every
number and both are explicit that a number is not advice. That is the whole
intersection.

## Where they differ

| | Edge Stats | crucible |
|---|---|---|
| Unit of analysis | a binary outcome per session: did the setup resolve, yes or no | a `TradeLog` in R-multiples: expectancy, payoff, profit factor, tail, not only the hit rate |
| The question | how often did X happen under condition Y | is this edge real after accounting for how it was found |
| Multiple testing | none. A query language over 42 presets x day-of-week x gap bucket x event flag is exactly the search-space explosion crucible corrects for, and the page never names it | `SearchSpaceLog` -> `variant_count()` -> deflated Sharpe and PBO. The honest N is the product |
| Uncertainty | Wilson interval on a proportion | seeded bootstrap interval on expectancy, permutation nulls, `reality_check` -> HELD / FRAGILE / FAIL |
| Out of sample | first half vs second half, no purge, no embargo | `holdout`, `walk_forward`, purge and embargo by construction |
| Time structure | per-year table, recency window | walk-forward, the DURABLE gate, and after promotion `edge_monitor` (CUSUM with a calibrated ARL0) |
| Breadth | none, one symbol at a time | `breadth`: effective independent bets across a correlated book |
| Verdict | none, you read the table | `run_gauntlet` REAL -> STRONG -> DURABLE -> GENERAL, hard checks, no override flag |
| Signal generation | yes, most of the product | no. crucible refuses to be the signal source |
| Data adapters | seven, plus broker imports | none, deliberately |
| Costs | not addressed; a win rate is cost-blind | R-denominated; costs are the caller's job but the log carries them |
| Platform | MCP, web, open-source engine | Python library, plus `pine/CrucibleEdge` for the scorecard half on TradingView |

The structural point: Edge Stats is built to **answer many questions fast**
(precompute, 20 ms, a query language, presets). crucible is built to **make it expensive
to fool yourself with one question**. Those pull in opposite directions. An interactive
query engine with no ledger of the queries run is a data-mining machine with a good
confidence interval on each result, and a per-result Wilson interval says nothing about
the two hundred queries discarded before the one that was kept.

The stability check is also weaker than it reads. `88.2% (n=51) vs 70.6% (n=51)` is
reported as "halves agree"; with n=51 each side the Wilson intervals are wide enough
that almost any two halves agree. crucible's equivalent (DURABLE, walk-forward
consistency) is a hard gate with numbers in `Thresholds`, not a tick mark.

## Where Edge Stats is ahead

- **Time to outcome.** Minutes to fill at p25 / p50 / p75 / p90. crucible carries
  `bars_held` and `exit_reason` on every `barrier_trades` log and reports only
  `time_asymmetry` (mean bars in winners over mean bars in losers) and a
  holding-vs-R scatter in the tearsheet. Nothing summarises how long the outcome takes.
  The sketch below closes that.
- **The MCP surface.** Read-only tools an assistant can call is a clean pattern.
  crucible has no equivalent, and whether it should is a separate question: a lens that
  answers an agent's queries on demand is one step from the agent running its own gate,
  which `AGENTS.md` forbids.
- **Broker fill import through one schema.** Nothing public in this stack does it.
- **Adapters and presets** lower the floor for someone who has no signal yet. crucible
  assumes you arrive with trades.

## How they would fit together

An Edge Stats outcome per session is a binary label under a condition. Turn each session
into a trade with a fixed target and stop (their gap-fill definition is close to a
target barrier already), express it in R, log every query you ran into a
`SearchSpaceLog`, and hand the result to `run_gauntlet`. That is the back half their
product does not have, and the front half crucible refuses to build.

One difference hides in that conversion. Edge Stats' "fill" is a target-only barrier
with a session-end timeout and no stop, so it has no risk unit and its result cannot be
denominated in R. Adding the stop is what turns a frequency into an edge, and it is also
what makes the number answer a different question.

**Bottom line:** not a competitor. Edge Stats is a fast, well-packaged
conditional-frequency lookup with honest per-query sample sizes and no correction for
the number of queries; crucible is the correction. If anything it demonstrates the gap
crucible fills: the tool invites exactly the interactive searching whose cost it does
not count.

## Design sketch: time to outcome in crucible

### What the log already carries

`barrier_trades` writes `bars_held` and `exit_reason` (`TP`, `SL`, `timeout`, `eod`)
on every trade, and `TradeLog` lists `bars_held` as an optional column that "time
metrics need". `TradeLog.from_frame` accepts an `exit_reason` column from any other
engine as an extra column. So the inputs exist; only the summary is missing.

### The quantity, and the trap in it

Edge Stats' percentiles are **conditional on the outcome having happened**: minutes to
fill, among sessions that filled. That is the natural number to show and the wrong one
to plan around, because a trade that has not resolved by the horizon is not slow, it is
**censored**. Reporting "median 4 bars to target" over the 60% of trades that hit the
target, while the other 40% timed out at bar 20, describes a fast edge that is mostly
not there.

The honest pair is therefore:

1. **Conditional percentiles per outcome class** (bars to TP among TP exits, bars to SL
   among SL exits), which is the Edge Stats number, labelled as conditional.
2. **The unconditional resolution curve**: the fraction of all trades resolved (either
   way) within k bars, which is what you can act on. With censoring only at a fixed
   horizon this is the empirical CDF of `bars_held` over resolved trades divided by N.
   `eod` exits censor early at varying k, so the general estimator is Kaplan-Meier,
   which reduces to that CDF when every censor sits at the horizon. It is a few lines of
   numpy and keeps the `edge` import surface at numpy + pandas.

The denominator is bars, never minutes or days. crucible has no clock, and a bar is
whatever the caller's frame is. Converting to calendar time is the caller's job, or
`crucible_stack.capital`'s.

### API

```python
from crucible.edge import time_to_outcome

tto = time_to_outcome(trades, q=(0.25, 0.5, 0.75, 0.9))
```

```python
@dataclass
class TimeToOutcome:
    n: int                      # every trade in the log
    n_resolved: int             # exit_reason in {TP, SL}, or every trade when the
                                # column is absent (then censored_share is 0 and
                                # `by_class` is keyed on the sign of r)
    censored_share: float       # timeout + eod, as a fraction of n
    horizon: Optional[int]      # max bars_held among timeout exits, None if no timeouts
    by_class: dict              # {"TP": {"n": 61, "q": {0.25: 2, 0.5: 4, 0.75: 7, 0.9: 12}},
                                #  "SL": {"n": 38, "q": {...}}}
    resolved_by: dict           # {1: 0.21, 2: 0.34, ..., horizon: 0.86}
                                # Kaplan-Meier fraction of ALL trades resolved within k bars
    q_all: dict                 # {0.25: 2, 0.5: 4, 0.75: 8, 0.9: None}, read off resolved_by;
                                # None where the curve never reaches the quantile
```

Rules the function keeps:

- `exit_reason` absent: fall back to the sign of `r` for the classes (`win` / `loss`),
  treat every trade as resolved, and say so in `censored_share = 0.0`. A log with no
  exit reasons cannot distinguish a slow loser from a timeout, and the result should
  not pretend otherwise.
- `bars_held` absent: return `None`, the same convention `edge_report` uses for
  `time_asymmetry`.
- Percentiles use `numpy.quantile(..., method="inverted_cdf")`: `bars_held` is an integer
  count and a p50 of 3.5 bars is not a thing anyone can wait for.
- Deterministic and free of randomness. This is a scorecard number, not a verdict, so it
  gets no seed and no bootstrap. If someone wants an interval on the median hold, the
  existing `bootstrap_ci` takes an arbitrary statistic.

### Where it surfaces

`EdgeReport` gains an optional `time_to_outcome: Optional[TimeToOutcome]`, populated by
`edge_report` when `bars_held` is present, and printed as one block:

```
Time to outcome (bars)      p25   p50   p75   p90
  TP       (n=61)             2     4     7    12
  SL       (n=38)             1     2     4     8
  all      (n=115)            2     4     8     -      [INFO]
  censored : 13.9 %  (16 timeout at 20 bars, 0 eod)
```

The `all` row is the same percentiles over every trade, read off the Kaplan-Meier
curve, and a `-` means the quantile is never reached before the horizon closes the
open trades. That dash is the point of the block: conditional on resolving, p90 is 12
bars; over the trades actually taken, one in ten is still open at bar 20, so p90 does
not exist. No horizon is chosen by the printer; the only horizon in the block is the
one the log carries.

#### Reading it

- **`TP` and `SL` rows: how fast each outcome arrives, given that it arrived.**
  `TP p75 = 7`: three quarters of the winners had hit the target by bar 7. `SL p50 = 2`:
  half the losers were stopped within 2 bars. Losers resolving faster than winners is the
  expected shape for a 2:1 target-to-stop (the stop is closer) and is what
  `time_asymmetry > 1` says with a single mean; the rows show the distribution behind it.
  Here the whole TP row sits to the right of the SL row, so the winners are not a few
  long rides dragging a mean up. The reverse shape, SL slower than TP, means the book is
  sitting in losers and the timeout is doing the stop's job.
- **`all` row: what you actually wait, per trade taken.** `p50 = 4`: half of all entries
  are closed within 4 bars. This row, not the class rows, is what capacity reads, because
  the class is unknown at entry. The `-` at p90 says no bar can be named by which 90% of
  trades are closed, because 13.9% were still open at the horizon. The q-th quantile of
  `all` is blank whenever the censored share exceeds 1 - q; at 30% censored, p75 blanks
  too.
- **`censored` line: how much the horizon shaped the log.** 16 of 115 trades exited at
  bar 20 on the clock, at whatever the close was. Their R is already in `expectancy`
  (they are `timeout`, neither win nor loss by construction). One in seven is enough to
  say the timeout is a live exit rule rather than a safety net, so its value is a
  parameter of the edge. `0 eod` says no trade was cut by the end of the data.
- **The question it raises and does not answer.** `TP p90 = 12` against a horizon of 20
  with 14% censored suggests some censored trades were winners still travelling, which
  makes "timeout 30" a tempting next run. That is a variant; it goes in the
  `SearchSpaceLog` and costs a unit of denominator like any other.
- **What it does not say.** Anything about whether the edge is real. `[INFO]` means no
  gate reads it.

The block is `[INFO]` only. Speed is not edge: a fast edge and a slow edge with the same
expectancy are the same edge to the judge. What speed changes is capacity, how many
positions the account holds at once, and that is `crucible_stack.capital`'s question,
answered from `concurrency_timeline`. Where speed does bear on the edge itself,
`time_asymmetry` already flags it, and the two read together: a winning class whose p75
sits past the losing class's p90 is `time_asymmetry > 1` with the shape shown.

The tearsheet gets a matching panel next to `holding_vs_r`: the resolution curve for
TP and SL as two step functions, with the censored share as the gap under 1.0 at the
horizon. That is the picture the Edge Stats page implies but does not draw.

### What it changes elsewhere

- **The Pine port.** The conditional percentiles are sorted-array lookups and port
  directly to `CrucibleEdge`; `tests/test_pine_conformance.py` will refuse the change
  until they do. The Kaplan-Meier curve is optional there (a chart script can draw the
  fixed-horizon CDF, which is the case `barrier_trades` produces) and the README's
  "what is deliberately not here" list should say which.
- **The docs.** `architecture.md`'s `TradeLog` column table gains `exit_reason` as the
  column time-to-outcome needs, and `tutorial.md`'s `edge_report` walk-through gets the
  block above with a one-line reading of it.
- **Nothing in `validation`.** No gate consults it, no threshold is added. It is not an
  input to REAL, STRONG, DURABLE or GENERAL, and adding it there would be the first
  step towards a "fast enough" check that has no honest null.

### What it does not do

It does not reproduce Edge Stats' minutes. A crucible log holds bars, and a bar on a
one-minute chart is a minute only until the caller resamples. It does not carry a
per-year breakdown; `walk_forward` is the time-consistency instrument here and a
per-year win-rate table is the thing it was built to replace. And it does not answer
"how often did the setup fill" for a target-only setup, because that log has no stop
and so no R; `barrier_trades` requires `sl`, and that requirement is the point.
