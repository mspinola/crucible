"""Runnable version of the README example — but on synthetic data so it needs no
network or the [examples] extra. Swap the `make_prices()` call for
`yfinance.download(...)` to run it on a real instrument.

    python examples/quickstart.py
"""
import numpy as np
import pandas as pd

from crucible.edge import barrier_trades, edge_report, reality_check, random_entry_null, expectancy
from crucible.strategies import ma_cross


def make_prices(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, 0.006, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def main():
    px = make_prices()
    entries = ma_cross(px, fast=20, slow=50)
    trades = barrier_trades(px, entries, side="long", tp=2.0, sl=1.0, timeout=20)

    print(edge_report(trades))
    print()
    print(reality_check(trades))

    null = random_entry_null(px, side="long", n_entries=trades.n, hold=20,
                             tp=2.0, sl=1.0, n_sims=500)
    pctile = float((null < expectancy(trades.r)).mean())
    print(f"\nRandom-entry null: signal beats {pctile:.0%} of coin-flip-timed books "
          f"(null mean E = {np.nanmean(null):+.3f} R).")


if __name__ == "__main__":
    main()
