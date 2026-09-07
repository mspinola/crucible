"""The published-system procedure from examples/stridsman_postpub.py, on REAL data.

Same rules, same frozen 1999 parameters, same judging: Stridsman's Standard
Deviation Breakout (2000), evaluated only on trades entered after the book was
published, one look in the ledger, the gauntlet with the detrended null scaled
into R. The difference is that nobody knows the verdict in advance here. Read
it as it comes out; the synthetic example is the one with a scripted ending.

Requires the [examples] extra and network access, so it is intentionally NOT part
of the test suite or CI:

    pip install "crucible[examples]"
    python examples/stridsman_postpub_yfinance.py                # ES=F, from 2000
    python examples/stridsman_postpub_yfinance.py --ticker SPY   # gap-free stand-in

Data caveats, disclosed up front, because they change what the numbers mean:

  * ``ES=F`` on Yahoo Finance is a front-month continuous series with the real
    calendar-spread GAP at every quarterly roll, not a back-adjusted contract. A
    percent stop or a band breakout can be triggered by a roll gap rather than by
    price. The faithful evaluation used ratio-adjusted continuous contracts, which
    the book itself insists on for percent-based statistics; Yahoo cannot supply
    them. Expect the gapped series to bias results against the system.
  * Yahoo's ES=F history starts in late 2000, so with a 60-bar band warmup the
    judged window effectively begins in 2001, not on the publication date.
  * ``SPY`` has no rolls and a longer history, at the cost of being an equity ETF
    rather than the futures contract the book traded.

Approximations against the book's execution model are the same as in the synthetic
example (close-confirmed band crossings; crucible's barrier exits at the book's
percent levels) and are documented there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from crucible.edge import edge_report

# Share one implementation of the rules and the judging with the synthetic example
# rather than carrying a second copy that could drift from it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stridsman_postpub import (  # noqa: E402
    BAND_K,
    BAND_LEN,
    PUB_DATE,
    STOP,
    TARGET,
    TIMEOUT,
    judge,
    write_report,
)


def load_ohlc(ticker: str, start: str) -> pd.DataFrame:
    """Daily OHLC from Yahoo Finance, normalized to plain Open/High/Low/Close
    columns on a DatetimeIndex (the shape `barrier_trades` expects)."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit('yfinance not installed; run:  pip install "crucible[examples]"')

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        sys.exit(f"no data returned for {ticker!r}; check the symbol / network.")
    if isinstance(df.columns, pd.MultiIndex):        # newer yfinance, single ticker
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df[["Open", "High", "Low", "Close"]].dropna()


def main() -> None:
    p = argparse.ArgumentParser(description="Stridsman's SDB on real post-publication data.")
    p.add_argument("--ticker", default="ES=F",
                   help="ES=F (front-month, roll gaps, see docstring) or SPY (gap-free)")
    p.add_argument("--start", default="2000-01-01",
                   help="download start; the judged window begins at the later of this "
                        "and the publication date, after indicator warmup")
    p.add_argument("--report", metavar="PATH",
                   help="also write the gauntlet HTML report here (needs crucible[report])")
    args = p.parse_args()

    px = load_ohlc(args.ticker, args.start)
    print(f"{args.ticker}: {len(px)} bars, {px.index.min().date()} -> {px.index.max().date()}")
    print(f"rules frozen at the book's values: EMA/sd bands {BAND_LEN}/{BAND_K}, "
          f"stop {STOP}%, target {TARGET}%, time {TIMEOUT} bars; nothing tuned here.\n")

    # one configuration, one look; a second ticker tried would be a second entry
    _, post, gauntlet = judge(px, scope=f"stridsman-sdb-postpub-{args.ticker}")
    if post.n < 30:
        sys.exit(f"only {post.n} post-publication trades; too few to judge anything.")
    print(f"POST-PUBLICATION, {post.n} trades entered on/after {PUB_DATE}:")
    print(edge_report(post))
    print("\n" + gauntlet.audit_report())
    print("\nGAUNTLET PASSED:", gauntlet.passed)
    if args.report:
        out = write_report(args.report, post, gauntlet,
                           title=f"Stridsman SDB (1999 parameters) on {args.ticker}",
                           subtitle=f"Yahoo Finance, {post.n} trades entered on/after "
                                    f"{PUB_DATE}, one look in the ledger; see the "
                                    "docstring for the roll-gap caveat")
        print(f"\nwrote {out}")
    print("\nThis is one instrument, one look, on a data series with the caveats in the "
          "docstring.\nWhatever it says, it says about this configuration on this series, "
          "with durability\nuntested (no walk-forward refit exists for frozen published "
          "parameters). It is not a\nverdict on the book, and a pass here would still "
          "need the full ladder before it meant\nanything.")


if __name__ == "__main__":
    main()
