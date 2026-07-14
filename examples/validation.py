"""Put a signal through the full crucible: holdout, walk-forward, permutation.
Synthetic data, no network — swap make_prices() for yfinance to use real data.

    python examples/validation.py
"""
import numpy as np
import pandas as pd

from crucible.edge import barrier_trades, reality_check
from crucible.strategies import ma_cross
from crucible.validation import holdout, walk_forward, sign_permutation_pvalue


def make_prices(n=2500, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, 0.006, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    idx = pd.date_range("2013-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def main():
    px = make_prices()
    trades = barrier_trades(px, ma_cross(px, 20, 50), side="long", tp=2.0, sl=1.0, timeout=20)

    print("1) POOLED reality check")
    print("  ", str(reality_check(trades)).replace("\n", "\n   "))

    print("\n2) EARLY/LATE HOLDOUT")
    print("  ", str(holdout(trades, "2018-01-01", embargo_weeks=8, n_boot=2000)).replace("\n", "\n   "))

    print("\n3) SIGN-PERMUTATION p-value")
    print(f"   p = {sign_permutation_pvalue(trades):.3f}")

    print("\n4) WALK-FORWARD (optimize fast/slow in-sample, confirm OOS)")
    wf = walk_forward(px, ma_cross, param_grid={"fast": [10, 20], "slow": [50, 100]},
                      is_days=365 * 3, oos_days=365, min_is_trades=5)
    print("  ", str(wf).replace("\n", "\n   "))
    print("\n   stitched OOS verdict:")
    print("  ", str(reality_check(wf.stitched)).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
