"""A generic, capital-free barrier simulator: OHLC + entry signal -> TradeLog in R.

No instrument specifics, no position sizing, no capital — just the geometry of
"enter here, exit at TP / SL / timeout" turned into R-multiple trades so the edge
metrics can read them. Look-ahead-free: barriers are sized off the SIGNAL bar
(known at entry), exits are scanned from the entry bar forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crucible.edge.trade_log import TradeLog

_OHLC = ("Open", "High", "Low", "Close")
_TRADE_COLS = ["r", "mfe", "mae", "bars_held", "entry_date", "exit_date", "exit_reason"]


def atr(df: pd.DataFrame, span: int = 14) -> pd.Series:
    """Wilder-ish ATR via EWM of true range."""
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=span, adjust=False).mean()


def barrier_trades(df: pd.DataFrame, entries, side: str = "long", *,
                   tp: float = 2.0, sl: float = 1.0, timeout: int = 20,
                   risk_unit: str = "atr", atr_col: str = "ATR",
                   entry: str = "next_open") -> TradeLog:
    """Enter on each True bar of `entries`; exit at TP / SL / timeout.

    tp, sl        barrier distances in `risk_unit`s (ATR multiples by default)
    risk_unit     'atr' (1 unit = ATR of the signal bar) or 'price' (1 unit = 1.0)
    entry         'next_open' (conservative) or 'close' (fill on the signal bar)

    1R = sl × unit, so returns pool cleanly across instruments. Positions never
    overlap: the scan resumes the bar after each exit.
    """
    missing = [c for c in _OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"barrier_trades needs OHLC columns; missing {missing}")

    df = df.copy()
    if risk_unit == "atr" and atr_col not in df.columns:
        df[atr_col] = atr(df)

    entries = pd.Series(entries).reindex(df.index, fill_value=False).to_numpy(dtype=bool)
    o, h, low, c = (df[x].to_numpy(dtype=float) for x in _OHLC)
    unit_arr = df[atr_col].to_numpy(dtype=float) if risk_unit == "atr" else np.ones(len(df))
    dates = df.index.to_numpy()
    s = 1 if side == "long" else -1
    n = len(df)

    trades = []
    i = 0
    while i < n - 1:
        if not entries[i]:
            i += 1
            continue
        ei = i + 1 if entry == "next_open" else i
        ep = o[ei] if entry == "next_open" else c[i]
        unit = unit_arr[i]                      # sized off the SIGNAL bar (no look-ahead)
        if not (ep > 0 and unit > 0):
            i += 1
            continue
        risk = sl * unit
        tp_px = ep + s * tp * unit
        sl_px = ep - s * sl * unit

        mfe = mae = 0.0
        xp = xr = None
        j = ei
        while j < n:
            fav = s * (h[j] - ep) if s > 0 else s * (low[j] - ep)
            adv = s * (low[j] - ep) if s > 0 else s * (h[j] - ep)
            mfe, mae = max(mfe, fav), min(mae, adv)
            if s > 0:
                if low[j] <= sl_px:
                    xp, xr = sl_px, "SL"
                    break
                if h[j] >= tp_px:
                    xp, xr = tp_px, "TP"
                    break
            else:
                if h[j] >= sl_px:
                    xp, xr = sl_px, "SL"
                    break
                if low[j] <= tp_px:
                    xp, xr = tp_px, "TP"
                    break
            if j - ei >= timeout:
                xp, xr = c[j], "timeout"
                break
            j += 1
        if xp is None:
            xp, xr, j = c[-1], "eod", n - 1

        trades.append({
            "r": s * (xp - ep) / risk,
            "mfe": mfe / risk, "mae": mae / risk,
            "bars_held": j - ei,
            "entry_date": dates[ei], "exit_date": dates[j], "exit_reason": xr,
        })
        i = j + 1                                # no overlapping positions

    frame = pd.DataFrame(trades, columns=_TRADE_COLS)
    return TradeLog(frame)


def random_entries(df: pd.DataFrame, n: int, seed: int = 0) -> pd.Series:
    """A boolean entry series with `n` random bars set True — the null model
    used by :func:`crucible.edge.stats.random_entry_null`."""
    rng = np.random.default_rng(seed)
    n = min(n, len(df) - 1)
    mask = np.zeros(len(df), dtype=bool)
    if n > 0:
        mask[rng.choice(len(df) - 1, size=n, replace=False)] = True
    return pd.Series(mask, index=df.index)
