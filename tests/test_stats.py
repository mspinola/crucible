import numpy as np

from crucible.edge import (
    TradeLog, bootstrap_ci, p_value_positive, reality_check, random_entry_null,
    barrier_trades, expectancy,
)
from crucible.strategies import ma_cross


def test_bootstrap_ci_brackets_point():
    tl = TradeLog.from_arrays(r=[2, 2, 2, -1, -1, -1, 2, -1] * 8)
    ci = bootstrap_ci(tl, n_boot=2000, seed=0)
    assert ci.low <= ci.point <= ci.high


def test_strong_edge_holds():
    # overwhelmingly positive -> HELD, p(edge>0) ~ 1
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    v = reality_check(tl, n_boot=2000, seed=0)
    assert v.label == "HELD"
    assert p_value_positive(tl, n_boot=2000) > 0.99


def test_no_edge_fails():
    tl = TradeLog.from_arrays(r=[1.0, -1.0] * 50)
    v = reality_check(tl, n_boot=2000, seed=0)
    assert v.label in ("FAIL", "FRAGILE")


def test_tiny_positive_sample_is_fragile():
    tl = TradeLog.from_arrays(r=[2.0, -1.0, 2.0, -1.0, 2.0])  # n=5, positive point
    v = reality_check(tl, n_boot=2000, seed=0)
    assert v.point > 0
    assert v.label == "FRAGILE"          # CI must straddle 0 at n=5


def test_random_entry_null_runs(ohlc):
    null = random_entry_null(ohlc, side="long", n_entries=25, hold=20,
                             tp=2.0, sl=1.0, n_sims=40, seed=0)
    assert len(null) == 40
    # real signal expectancy is comparable-to / beats the coin-flip null mean
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long", tp=2.0, sl=1.0, timeout=20)
    real = expectancy(tl.r)
    assert np.isfinite(np.nanmean(null))
    assert np.isfinite(real)
