"""Generate the committed sample RealTest export (``ma_cross_trades.csv``).

A maintainer helper, NOT needed to run the example. It fabricates a realistic
`SaveTradesAs` CSV so ``examples/realtest_ingest.py`` runs offline with no
RealTest license. The trades come from the *same* rule the paired
``ma_cross.rts`` describes, run on a reproducible synthetic price series:

  * enter long the bar after the fast SMA crosses **above** the slow SMA,
  * exit the bar after it crosses back **below**, or intrabar at a fixed
    ``STOP_PCT`` protective stop (that stop distance is what makes 1R well
    defined: 1R == STOP_PCT, so R = PctGain / STOP_PCT).

Columns and formatting mirror RealTest's stock trade export (percent fields
carry a trailing ``%``, dates are ``YYYY-MM-DD``). Regenerate with::

    python examples/realtest/make_sample_export.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FAST, SLOW = 20, 50
STOP_PCT = 5.0          # protective stop, in %. This IS 1R for the export.
SYMBOL = "SPY"
STRATEGY = "MACrossLong"
SIZE = 10_000.0         # $ position size (fixed-fractional stand-in)
OUT = Path(__file__).resolve().parent / "ma_cross_trades.csv"


def synthetic_ohlc(n: int = 8200, seed: int = 11) -> pd.DataFrame:
    """A trending-but-choppy daily series, enough 20/50 crossings for ~100 trades."""
    rng = np.random.default_rng(seed)
    # mild upward drift with frequent regime wobble so the MAs cross both ways often
    drift = 0.0003 + 0.0008 * np.sin(np.linspace(0, 42 * np.pi, n))
    rets = rng.normal(drift, 0.012, n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, 0.007, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    idx = pd.date_range("1993-01-29", periods=n, freq="B")   # SPY's inception era
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def run_ma_cross(px: pd.DataFrame) -> pd.DataFrame:
    fast = px["Close"].rolling(FAST).mean()
    slow = px["Close"].rolling(SLOW).mean()
    up = (fast > slow) & (fast.shift(1) <= slow.shift(1))     # golden cross
    dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))     # death cross

    rows, in_pos = [], False
    entry_i = entry_px = stop_px = None
    n = len(px)
    for i in range(SLOW + 1, n):
        if not in_pos and up.iloc[i] and i + 1 < n:
            in_pos, entry_i = True, i + 1                      # fill next bar's open
            entry_px = float(px["Open"].iloc[entry_i])
            stop_px = entry_px * (1 - STOP_PCT / 100)
            continue
        if in_pos:
            lo = float(px["Low"].iloc[i])
            hit_stop = lo <= stop_px
            cross_out = dn.iloc[i]
            if hit_stop or cross_out or i == n - 1:
                exit_i = i if hit_stop else min(i + 1, n - 1)  # stop fills intrabar
                exit_px = stop_px if hit_stop else float(px["Open"].iloc[exit_i])
                win = px.iloc[entry_i:exit_i + 1]
                mfe = (float(win["High"].max()) / entry_px - 1) * 100
                mae = (float(win["Low"].min()) / entry_px - 1) * 100
                pct = (exit_px / entry_px - 1) * 100
                rows.append({
                    "DateIn": px.index[entry_i].date(), "PriceIn": entry_px,
                    "DateOut": px.index[exit_i].date(), "PriceOut": exit_px,
                    "Reason": "Stop" if hit_stop else "Signal",
                    "Bars": exit_i - entry_i,
                    "PctGain": pct, "PctMFE": mfe, "PctMAE": mae,
                })
                in_pos = False
    return pd.DataFrame(rows)


def to_realtest_csv(t: pd.DataFrame) -> pd.DataFrame:
    qty = (SIZE / t["PriceIn"]).round().astype(int)
    pct = lambda s: s.map(lambda v: f"{v:.2f}%")
    return pd.DataFrame({
        "Trade": range(1, len(t) + 1),
        "Strategy": STRATEGY,
        "Symbol": SYMBOL,
        "Side": "Long",
        "DateIn": t["DateIn"], "TimeIn": "00:00:00",
        "QtyIn": qty, "PriceIn": t["PriceIn"].round(2),
        "DateOut": t["DateOut"], "TimeOut": "00:00:00",
        "QtyOut": qty, "PriceOut": t["PriceOut"].round(2),
        "Reason": t["Reason"], "Bars": t["Bars"],
        "PctGain": pct(t["PctGain"]),
        "Profit": (t["PctGain"] / 100 * SIZE).round(2),
        "PctMFE": pct(t["PctMFE"]), "PctMAE": pct(t["PctMAE"]),
        "Fraction": f"{SIZE / 100_000:.4f}", "Size": round(SIZE, 2), "Dividends": "0.00",
    })


def main() -> None:
    trades = run_ma_cross(synthetic_ohlc())
    out = to_realtest_csv(trades)
    out.to_csv(OUT, index=False)
    wr = trades["PctGain"].gt(0).mean()
    print(f"wrote {OUT.name}: {len(out)} trades, win rate {wr*100:.1f}%, "
          f"mean PctGain {trades['PctGain'].mean():+.2f}% (1R = {STOP_PCT:.0f}%)")


if __name__ == "__main__":
    main()
