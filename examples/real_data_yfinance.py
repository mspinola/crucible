"""Run the full crucible pipeline on REAL market data pulled from Yahoo Finance.

Requires the [examples] extra:

    pip install "crucible-quant[examples]"
    python examples/real_data_yfinance.py                 # SPY, 20/50 MA cross
    python examples/real_data_yfinance.py --ticker QQQ --fast 10 --slow 30

This needs network access, so it's intentionally NOT part of the test suite /
CI — the synthetic examples/*.py are the smoke tests. It exists to show crucible
reading a real OHLC frame and to let you kick a signal you actually care about.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from crucible.edge import barrier_trades, edge_report, reality_check
from crucible.strategies import ma_cross
from crucible.validation import holdout, walk_forward, sign_permutation_pvalue


def load_ohlc(ticker: str, start: str) -> pd.DataFrame:
    """Download a daily OHLC frame and normalize it to what crucible expects:
    a DatetimeIndex with plain `Open/High/Low/Close` columns, split-adjusted."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit('yfinance not installed — run:  pip install "crucible-quant[examples]"')

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        sys.exit(f"no data returned for {ticker!r} — check the symbol / network.")
    # Newer yfinance returns MultiIndex columns (('Open','SPY'), ...) even for a
    # single ticker; flatten to the price level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df[["Open", "High", "Low", "Close"]].dropna()


def main() -> None:
    p = argparse.ArgumentParser(description="crucible on real Yahoo Finance data.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--split", default="2018-01-01", help="holdout early/late boundary")
    p.add_argument("--side", default="long", choices=["long", "short"])
    args = p.parse_args()

    px = load_ohlc(args.ticker, args.start)
    print(f"{args.ticker}: {len(px)} bars, {px.index.min().date()} -> {px.index.max().date()}")

    entries = ma_cross(px, fast=args.fast, slow=args.slow)
    trades = barrier_trades(px, entries, side=args.side, tp=2.0, sl=1.0, timeout=20)
    print(f"{args.fast}/{args.slow} MA cross ({args.side}): {trades.n} trades\n")

    print(edge_report(trades))
    print("\n1) POOLED reality check")
    print("  ", str(reality_check(trades)).replace("\n", "\n   "))

    print("\n2) EARLY/LATE HOLDOUT")
    print("  ", str(holdout(trades, args.split, embargo_weeks=8, n_boot=3000)).replace("\n", "\n   "))

    print(f"\n3) SIGN-PERMUTATION p-value: {sign_permutation_pvalue(trades):.3f}")

    print("\n4) WALK-FORWARD (optimize fast/slow in-sample, confirm OOS)")
    wf = walk_forward(px, ma_cross,
                      param_grid={"fast": [10, 20, 30], "slow": [50, 100, 200]},
                      side=args.side, is_days=365 * 4, oos_days=365, min_is_trades=5)
    print("  ", str(wf).replace("\n", "\n   "))
    print("\n   stitched OOS verdict:")
    print("  ", str(reality_check(wf.stitched)).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
