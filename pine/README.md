# CrucibleEdge: crucible.edge on TradingView

A function-for-function Pine v6 port of `crucible.edge`, so a number on a chart means what
the same number means in crucible's edge report. Same names, same formulas, same
conventions (infinity becomes `na` plus an `isInf` predicate; `None` becomes `na`).

| File | What it is |
|---|---|
| `CrucibleEdge.pine` | the library: `TradeLog`, every edge metric, `edgeReport`, cumulative R and max drawdown, CSV export for `TradeLog.from_frame`, and the report drawn as a themed table (list or grid) with crucible's own flags and definitions as tooltips |
| `CrucibleEdgeConformance.pine` | an indicator that runs every function on fixed logs and compares against values crucible produced. Add it to any chart after publishing the library: the table must read ALL PASS before anything downstream is trusted |
| `gen_vectors.py` | produces those values from crucible itself; `--write` refreshes `vectors.json` |
| `vectors.json` | the committed vectors, with the crucible version that made them |

`tests/test_pine_conformance.py` closes the loop from this side: it regenerates the vectors
on every run and fails when `vectors.json` or the numbers embedded in the conformance script
differ from what the current crucible computes. A change to an edge metric therefore cannot
land without the Pine port being updated and republished.

**What is deliberately not here:** `bootstrap_ci`, `reality_check`, the nulls, `pbo`,
`run_gauntlet`, `SearchSpaceLog`. Those need seeded randomness and the count of every variant
searched, which a chart script cannot know. The library is the scorecard, not the judge.
Export the log with `csvLine` and run the gate in Python.

## Publishing

TradingView compiles Pine; nothing here does. Publish `CrucibleEdge` as a private library,
then add `CrucibleEdgeConformance` (with its import line set to the version assigned) to a
chart and confirm ALL PASS. Scripts that consume the library through another library (for
example a trade recorder holding a `CrucibleEdge.TradeLog`) must import the exact same
CrucibleEdge version that library was published against, or Pine refuses the type.

## Updating after a metric change

1. `python pine/gen_vectors.py --write`
2. Copy the new expected values into the `mk(...)` calls in `CrucibleEdgeConformance.pine`
   (the test says which case and metric moved).
3. Port the change into `CrucibleEdge.pine`, republish, rerun the conformance indicator.
