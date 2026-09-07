"""Judging a PUBLISHED trading system on post-publication data.

Case study: Thomas Stridsman's *Trading Systems That Work* (2000) froze every
parameter of its systems on data ending October 1999. That makes the publication
date a free holdout: everything after it is data the author's search never saw,
and the only search left to correct for is yours. In 2026 I ran the book's final
configurations through this procedure on 26 years of real futures (ratio-adjusted,
costed, pre-registered): five of six showed nothing, and the Standard Deviation
Breakout kept a small real lean (~+0.14R/trade) that still failed the gauntlet on
effect size and cross-market generality, which is roughly what the book's own
philosophy predicts for parameters frozen in 1999.

This example demonstrates that procedure end to end on SYNTHETIC prices: no real
asset is involved, no network, exact numbers every run. The series is a seeded
random walk built to tell the story: multi-year trend regimes are planted BEFORE
the publication date (the era the rules were fitted to) and none after, so the
full-history scorecard flatters and the post-publication gauntlet does not. For
the same procedure on a real instrument, see ``examples/stridsman_postpub_yfinance.py``.

  1. implement the published rules with the published parameters, no tuning;
  2. split at the "publication date" and evaluate only trades entered after it;
  3. count every configuration you looked at in a ``SearchSpaceLog``; even one
     look is a look, and the ledger is what keeps the correction honest;
  4. run the gauntlet with the detrended null and an R-denominated ``null_scale``
     (a mixed long/short book on drift-bearing prices needs both; skipping the
     scale compares an R expectancy against a fractional-return bar that almost
     any positive book clears).

Approximations vs. the book, disclosed: entries are close-confirmed crossings of
the band (the book works intrabar stop orders at the band), and the exit suite is
crucible's stop/target/timeout barriers at the book's percent levels (the book
adds a channel exit and a trailing stop). The faithful reproduction needs a
bespoke intrabar simulator; this file is about the judging, not the simulating.

    python examples/stridsman_postpub.py
"""
import numpy as np
import pandas as pd

from crucible.edge import TradeLog, barrier_trades, edge_report
from crucible.validation import SearchSpaceLog, Thresholds, run_gauntlet

PUB_DATE = "2000-01-03"          # first bar the author's search never saw
BAND_LEN, BAND_K = 60, 2.0       # the book's SDB bands: EMA of highs/lows ± 2 sd
STOP, TARGET, TIMEOUT = 5.5, 75.0, 105   # its Ch. 10 exit suite, percent of entry


def synthetic_prices(n: int = 8600, seed: int = 11) -> pd.DataFrame:
    """~33y of daily OHLC starting 1993. Multi-year drift regimes are planted
    only BEFORE PUB_DATE (the era the published rules were fitted to); after it
    the series is a driftless random walk, the trends the rules depended on
    gone. Long enough that indicators are warm well before PUB_DATE."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1993-01-04", periods=n, freq="B")
    regimes = 0.0003 + 0.0016 * np.sin(np.arange(n) / 600)
    drift = np.where(idx < pd.Timestamp(PUB_DATE), regimes, 0.0)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.010, n))
    op = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, 0.005, n)) * close
    return pd.DataFrame(
        {"Open": op, "High": np.maximum(op, close) + span,
         "Low": np.minimum(op, close) - span, "Close": close},
        index=idx,
    )


def sdb_bands(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The book's Standard Deviation Breakout bands, built on the HIGHS and
    LOWS, not closes: upper = EMA(High, 60) + 2·sd(High, 60), lower mirrored on
    the lows. Deliberately unoptimized in the book, frozen here the same way."""
    upper = (df["High"].ewm(span=BAND_LEN, adjust=False).mean()
             + BAND_K * df["High"].rolling(BAND_LEN).std())
    lower = (df["Low"].ewm(span=BAND_LEN, adjust=False).mean()
             - BAND_K * df["Low"].rolling(BAND_LEN).std())
    return upper, lower


def sdb_trades(px: pd.DataFrame) -> TradeLog:
    """Both sides of the SDB under percent-denominated barrier exits.

    ``risk_unit`` trick: a column holding 1% of the close makes ``tp``/``sl``
    read as percent-of-price, so sl=5.5 risks 5.5% and 1R = that risk, the
    book's own unit convention, poolable across markets and decades."""
    px = px.copy()
    px["PCT"] = px["Close"] * 0.01
    upper, lower = sdb_bands(px)
    go_long = (px["Close"] > upper.shift(1)) & (px["Close"].shift(1) <= upper.shift(2))
    go_short = (px["Close"] < lower.shift(1)) & (px["Close"].shift(1) >= lower.shift(2))

    frames = []
    for entries, side, direction in ((go_long, "long", 1.0), (go_short, "short", -1.0)):
        log = barrier_trades(px, entries, side=side, tp=TARGET, sl=STOP,
                             timeout=TIMEOUT, risk_unit="atr", atr_col="PCT")
        frame = log.frame.copy()
        frame["direction"] = direction
        frames.append(frame)
    both = pd.concat(frames, ignore_index=True).sort_values("entry_date")
    return TradeLog(both.reset_index(drop=True))


THR = Thresholds(n_boot=5000, n_perm=5000, n_random_sims=500)


def judge(px: pd.DataFrame, *, scope: str = "stridsman-sdb-postpub", thr: Thresholds = THR):
    """The whole procedure on one OHLC frame: (full-history log, post-publication
    log, gauntlet verdict). One code path, shared by this example, the yfinance
    example, and the test that guards the narrative."""
    # 1-2. published rules; only trades entered after publication are judged
    all_trades = sdb_trades(px)
    post = TradeLog(all_trades.frame[all_trades.frame["entry_date"] >= PUB_DATE]
                    .reset_index(drop=True))

    # 3. the honest denominator: every configuration looked at goes in the ledger.
    # One frozen configuration here; the real 2026 evaluation carried nine
    # (seven book-final configurations plus two exploratory looks; looks count).
    looks = SearchSpaceLog(scope=scope)
    looks.record({"band_len": BAND_LEN, "band_k": BAND_K,
                  "stop": STOP, "target": TARGET, "timeout": TIMEOUT},
                 status="tried")

    # 4. the gauntlet. A mixed long/short R-denominated book needs the detrended
    # null with per-trade directions, and null_scale = entry/risk to put the null
    # in R too (risk is 5.5% of price here, so entry/risk = 1/0.055 per trade).
    post_px = px[px.index >= PUB_DATE]
    hold = int(round(float(post.frame["bars_held"].mean())))
    gauntlet = run_gauntlet(
        post, prices=post_px, hold=hold,
        null="detrended",
        directions=post.frame["direction"].to_numpy(),
        null_scale=np.full(post.n, 1.0 / (STOP / 100.0)),
        n_variants=looks,
        thr=thr,
    )
    return all_trades, post, gauntlet


def main() -> None:
    px = synthetic_prices()
    all_trades, post, gauntlet = judge(px)

    # the full history flatters (its trends were planted for the rules to find)
    print(f"FULL HISTORY, {all_trades.n} trades (the era the rules were built for):")
    print(edge_report(all_trades))
    print(f"\nPOST-PUBLICATION ONLY, {post.n} trades entered on/after {PUB_DATE} "
          "(the only ones judged):")
    print(edge_report(post))
    print("\n" + gauntlet.audit_report())
    print("\nGAUNTLET PASSED:", gauntlet.passed)
    print("\nThe full-history scorecard looked fine because the trends it fed on "
          "were planted\nbefore publication; on the years after, the gate refuses "
          "it. That is the machinery\nworking: a published system is owed nothing. "
          "On the real 26 post-publication years,\nthe book's systems eroded to "
          "roughly nothing, quietly, without blowing up, exactly\nthe failure mode "
          "its own robustness philosophy was designed to produce.")


if __name__ == "__main__":
    main()
