"""Conformance vectors for the CrucibleEdge Pine library, produced by crucible itself.

Fixed synthetic trade logs (including crucible's edge cases) and every metric
`crucible.edge.edge_report` reports on them, plus the cumulative-R max drawdown the
tearsheet draws. `pine/vectors.json` is the committed output and the numbers are embedded
in `pine/CrucibleEdgeConformance.pine`; `tests/test_pine_conformance.py` regenerates the
vectors on every run and fails when either file has drifted from the library.

Deterministic: no randomness, no store access, nothing but crucible.

Usage:  python pine/gen_vectors.py            # prints the JSON
        python pine/gen_vectors.py --write    # rewrites pine/vectors.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import crucible
from crucible.edge import TradeLog, edge_report

VECTORS_PATH = Path(__file__).with_name("vectors.json")
CONFORMANCE_PATH = Path(__file__).with_name("CrucibleEdgeConformance.pine")

CASES = {
    # mixed winners, losers, one scratch, mfe/mae/bars present
    "mixed": dict(r=[2.5, -1.0, 0.8, -1.0, 4.2, -0.6, 0.0, 1.1, -1.0, 3.0, -0.4, 0.5],
                  mfe=[3.1, 0.4, 1.2, 0.2, 5.0, 0.9, 0.7, 1.5, 0.1, 3.6, 0.3, 1.0],
                  mae=[-0.3, -1.0, -0.5, -1.0, -0.2, -1.0, -0.4, -0.6, -1.0, -0.1, -1.0, -0.8],
                  bars=[12, 3, 7, 2, 30, 4, 5, 9, 1, 25, 3, 6]),
    # no losers: profit factor and payoff are +inf
    "no_losers": dict(r=[1.0, 2.0, 0.5], mfe=[1.2, 2.4, 0.9], mae=[-0.2, -0.1, -0.3], bars=[4, 8, 2]),
    # all losers: also pins the drawdown convention (peak starts at the first trade, so 1.5 not 2.5)
    "all_losers": dict(r=[-1.0, -0.5, -1.0], mfe=[0.2, 0.4, 0.0], mae=[-1.0, -0.8, -1.0], bars=[2, 5, 1]),
    # identical returns: sqn guard -> 0
    "identical": dict(r=[0.7, 0.7, 0.7, 0.7], mfe=[1.0, 1.0, 1.0, 1.0],
                      mae=[-0.2, -0.2, -0.2, -0.2], bars=[3, 3, 3, 3]),
    # single trade: sqn 0 (n < 2)
    "single": dict(r=[1.5], mfe=[2.0], mae=[-0.4], bars=[6]),
    # more than 100 trades: sqn cap
    "n150": dict(r=[((i * 7919) % 13 - 5) / 4.0 for i in range(150)], mfe=None, mae=None, bars=None),
}

# The metrics the Pine port reports, in the order the conformance script's `mk(...)` takes them.
METRICS = ("n", "win_rate", "expectancy", "profit_factor", "payoff_ratio", "sqn",
           "excursion_ratio", "time_asymmetry", "exit_efficiency", "max_drawdown_r")


def maxdd(r) -> float:
    """report.tearsheet.equity_drawdown's number: deepest peak-to-trough of summed R, the
    running peak starting at the first trade's cumulative R (not at 0)."""
    eq = np.cumsum(np.asarray(r, float))
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


def enc(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return x


def vectors() -> dict:
    out = {"crucible_version": crucible.__version__, "cases": {}}
    for name, c in CASES.items():
        kw = {k: v for k, v in (("mfe", c["mfe"]), ("mae", c["mae"]), ("bars_held", c["bars"]))
              if v is not None}
        log = TradeLog.from_arrays(c["r"], **kw)
        rep = edge_report(log).to_dict()
        rep.pop("calibration", None)
        rep["max_drawdown_r"] = maxdd(c["r"])
        out["cases"][name] = {"inputs": c, "expected": {k: enc(v) for k, v in rep.items()}}
    return out


def main(argv: list[str]) -> int:
    text = json.dumps(vectors(), indent=1)
    if "--write" in argv:
        VECTORS_PATH.write_text(text + "\n")
        print(f"wrote {VECTORS_PATH}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
