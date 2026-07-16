"""Worked example for the tutorial (docs/index.md §12; https://mspinola.github.io/crucible/).

A Donchian channel breakout run end to end through crucible: describe the edge,
reality-check it, walk it forward, and put it through the gauntlet. Uses
reproducible synthetic prices (no network) so it prints the exact numbers the
tutorial walks through — a breakout that is real and strong in aggregate (REAL ✓,
STRONG ✓) yet fails DURABLE, so the gauntlet rejects it.

    python examples/donchian_gauntlet.py
"""
import numpy as np
import pandas as pd

from crucible.edge import barrier_trades, edge_report, reality_check
from crucible.validation import walk_forward, run_gauntlet, Thresholds


def donchian(df: pd.DataFrame, lookback: int = 20, price: str = "Close") -> pd.Series:
    """Long entry when Close breaks above the prior `lookback`-bar high."""
    return df[price] > df["High"].rolling(lookback).max().shift(1)


def synthetic_prices(n: int = 4200, seed: int = 7) -> pd.DataFrame:
    """~16y of daily OHLC: multi-year bull/bear regimes over a mild uptrend, so a
    breakout catches real trends in some eras and whipsaws in others."""
    rng = np.random.default_rng(seed)
    drift = 0.0005 + 0.0016 * np.sin(np.arange(n) / 500)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.009, n))
    op = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, 0.005, n)) * close
    return pd.DataFrame(
        {"Open": op, "High": np.maximum(op, close) + span,
         "Low": np.minimum(op, close) - span, "Close": close},
        index=pd.date_range("2008-01-01", periods=n, freq="B"),
    )


def main() -> None:
    px = synthetic_prices()
    TP, SL, TIMEOUT = 2.5, 1.0, 30

    # 1. the signal -> a trade log, and the capital-free scorecard
    trades = barrier_trades(px, donchian(px, 20), side="long", tp=TP, sl=SL, timeout=TIMEOUT)
    print(edge_report(trades))

    # 2. is the pooled edge real, or small-sample luck?
    print("\n" + str(reality_check(trades)))

    # 3. does it survive out of sample, over time?
    wf = walk_forward(px, donchian, param_grid={"lookback": [20, 40]},
                      is_days=365 * 3, oos_days=365, tp=TP, sl=SL, timeout=TIMEOUT)
    print("\n" + str(wf))

    # 4. the verdict — the whole gauntlet, capital-free
    gauntlet = run_gauntlet(wf.stitched, prices=px, wf=wf, side="long", tp=TP, sl=SL,
                            n_variants=2, thr=Thresholds(n_boot=5000, n_perm=5000, n_random_sims=500))
    print("\n" + gauntlet.audit_report())
    print("\nGAUNTLET PASSED:", gauntlet.passed)


if __name__ == "__main__":
    main()
