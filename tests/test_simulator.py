import numpy as np

from crucible.edge import barrier_trades, random_entries, TradeLog
from crucible.strategies import ma_cross, macd_cross


def test_barrier_trades_shapes(ohlc):
    entries = ma_cross(ohlc, fast=20, slow=50)
    tl = barrier_trades(ohlc, entries, side="long", tp=2.0, sl=1.0, timeout=20)
    assert isinstance(tl, TradeLog)
    assert tl.n > 0
    for c in ("r", "mfe", "mae", "bars_held", "exit_reason"):
        assert c in tl.frame.columns
    # SL fills at the stop, so stopped trades book exactly -1R; MAE is the
    # intrabar excursion and may dip past -1R when a wide bar gaps through.
    sl_r = tl.frame.loc[tl.frame["exit_reason"] == "SL", "r"]
    assert np.allclose(sl_r, -1.0)
    assert (tl.frame["mae"] <= 1e-9).all()       # adverse excursion is non-positive
    assert (tl.frame["mfe"] >= -1e-9).all()       # favorable excursion is non-negative


def test_no_overlapping_positions(ohlc):
    entries = macd_cross(ohlc)
    tl = barrier_trades(ohlc, entries, side="long")
    f = tl.frame.sort_values("entry_date")
    # each entry is at or after the previous exit
    assert (f["entry_date"].values[1:] >= f["exit_date"].values[:-1]).all()


def test_price_risk_unit(ohlc):
    entries = ma_cross(ohlc)
    tl = barrier_trades(ohlc, entries, side="long", tp=4.0, sl=2.0, risk_unit="price")
    assert tl.n > 0


def test_random_entries_count(ohlc):
    e = random_entries(ohlc, 30, seed=1)
    assert int(e.sum()) == 30
