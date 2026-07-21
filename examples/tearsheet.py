"""Generate a self-contained HTML tearsheet for a signal. Synthetic data by
default (no network); requires the [report] extra for plotly.

    pip install "crucible[report]"
    python examples/tearsheet.py                       # -> tearsheet.html (synthetic)
    python examples/tearsheet.py --ticker SPY --out spy.html   # real data, needs [examples]
"""
import argparse

import numpy as np
import pandas as pd

from crucible.edge import barrier_trades
from crucible.strategies import ma_cross
from crucible.report import tearsheet


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
    p = argparse.ArgumentParser(description="Render a crucible tearsheet.")
    p.add_argument("--ticker", default=None, help="Yahoo symbol (needs the [examples] extra); omit for synthetic data.")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--out", default="tearsheet.html")
    args = p.parse_args()

    if args.ticker:
        from real_data_yfinance import load_ohlc
        px = load_ohlc(args.ticker, "2005-01-01")
        title = f"{args.ticker} — {args.fast}/{args.slow} MA cross (long)"
    else:
        px = make_prices()
        title = f"Synthetic — {args.fast}/{args.slow} MA cross (long)"

    trades = barrier_trades(px, ma_cross(px, args.fast, args.slow),
                            side="long", tp=2.0, sl=1.0, timeout=20)
    path = tearsheet(trades, args.out, title=title,
                     subtitle=f"{trades.n} trades · 2R/1R/20-bar barriers")
    print(f"wrote {path}  ({trades.n} trades)")


if __name__ == "__main__":
    main()
