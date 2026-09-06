# The Pine port

`pine/CrucibleEdge.pine` is a function-for-function port of `crucible.edge` for
TradingView, so a chart script can score its own trade log with the same metrics, the same
conventions, and the same flags crucible's edge report prints, and draw them as a table.
It is the scorecard only: nothing from the gauntlet is ported, because a chart cannot know
the size of the search that produced a strategy, and without that denominator no
significance claim is honest.

Two checks keep the port and crucible in step:

- On a chart, `pine/CrucibleEdgeConformance.pine` runs every function on fixed logs and
  compares against values crucible itself produced. It must read ALL PASS after a publish.
- In this repo, `tests/test_pine_conformance.py` regenerates those values from the current
  crucible and fails when the committed vectors or the numbers embedded in the conformance
  script have drifted. A metric change cannot land without the port following.

Details, the publishing order, and what to do after a metric change are in
[`pine/README.md`](https://github.com/mspinola/crucible/blob/main/pine/README.md).

One convention worth knowing because it was caught by exactly this check: the max drawdown
of summed R starts its running peak at the first trade's cumulative R, as
`report.tearsheet.equity_drawdown` does, not at the flat start of zero. A log that opens with
losers therefore reports the drawdown from its first close.
