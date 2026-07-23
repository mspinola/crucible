import numpy as np

from crucible.edge import (
    TradeLog,
    barrier_trades,
    block_bootstrap_ci,
    block_bootstrap_pvalue,
    bootstrap_ci,
    bootstrap_metric_cis,
    expectancy,
    p_value_positive,
    profit_factor,
    random_entry_null,
    reality_check,
    sqn,
    win_rate,
)
from crucible.strategies import ma_cross


def test_bootstrap_ci_brackets_point():
    tl = TradeLog.from_arrays(r=[2, 2, 2, -1, -1, -1, 2, -1] * 8)
    ci = bootstrap_ci(tl, n_boot=2000, seed=0)
    assert ci.low <= ci.point <= ci.high


def test_bootstrap_metric_cis_default_set_and_points_match():
    tl = TradeLog.from_arrays(r=[2, 2, -1, -1, 2, -1, 2, -1] * 10)
    cis = bootstrap_metric_cis(tl, n_boot=2000, seed=0)
    assert set(cis) == {"expectancy", "profit_factor", "sqn", "win_rate"}
    r = tl.r
    # each point estimate equals the metric on the full sample, and the CI brackets it
    for name, fn in (("expectancy", expectancy), ("profit_factor", profit_factor),
                     ("sqn", sqn), ("win_rate", win_rate)):
        assert cis[name].point == fn(r)
        assert cis[name].low <= cis[name].point <= cis[name].high


def test_bootstrap_metric_cis_strong_edge_clears_zero():
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    cis = bootstrap_metric_cis(tl, n_boot=3000, seed=0)
    assert cis["expectancy"].low > 0          # lower bound clears zero -> HELD-worthy
    assert cis["profit_factor"].low > 1.0


def test_bootstrap_metric_cis_custom_metric_set():
    tl = TradeLog.from_arrays(r=[1.0, -1.0, 2.0, -1.0] * 20)
    cis = bootstrap_metric_cis(tl, metrics={"win_rate": win_rate}, n_boot=1000, seed=0)
    assert set(cis) == {"win_rate"}


def test_bootstrap_metric_cis_empty_is_nan():
    cis = bootstrap_metric_cis(TradeLog.from_arrays(r=[]), n_boot=500)
    assert np.isnan(cis["expectancy"].point)
    assert np.isnan(cis["expectancy"].low)


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


# --- block bootstrap (significance under serial dependence) ------------------



def _ci_width(ci):
    return ci.high - ci.low


def test_block_bootstrap_positive_edge_low_p():
    r = np.random.default_rng(0).normal(0.3, 1.0, 400)
    assert block_bootstrap_pvalue(r, block=1, n_boot=5000) < 0.05


def test_block_bootstrap_zero_edge_not_significant():
    r = np.random.default_rng(1).normal(0.0, 1.0, 400)
    r = r - r.mean()                                   # exactly zero mean -> p ~ 0.5
    assert block_bootstrap_pvalue(r, block=1, n_boot=5000) > 0.3
    assert block_bootstrap_pvalue(r + 0.3, block=1, n_boot=5000) < 0.05


def test_block_absorbs_positive_autocorrelation():
    # AR(1), strong positive autocorrelation + positive mean: the i.i.d. (block=1)
    # test overstates significance; a longer block absorbs the autocorrelation ->
    # higher p and a wider CI. This is the whole point vs the trade-level bootstrap.
    rng = np.random.default_rng(2)
    n, rho = 500, 0.85
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0.0, 1.0)
    x = x - x.mean() + 0.25            # pin a positive sample mean; keep the AR(1) structure
    p1 = block_bootstrap_pvalue(x, block=1, n_boot=5000)
    p20 = block_bootstrap_pvalue(x, block=20, n_boot=5000)
    assert p20 > p1                                    # block raises p (less significant)
    assert _ci_width(block_bootstrap_ci(x, block=20, n_boot=5000)) > \
           _ci_width(block_bootstrap_ci(x, block=1, n_boot=5000))   # and widens the CI


def test_block_bootstrap_stationary_option():
    r = np.random.default_rng(3).normal(0.25, 1.0, 400)
    assert block_bootstrap_pvalue(r, block=6, n_boot=4000, stationary=True) < 0.05
    assert block_bootstrap_pvalue(r, block=6, n_boot=4000, stationary=False) < 0.05


def test_block_bootstrap_short_series_guards():
    assert block_bootstrap_pvalue([0.5], block=3) == 1.0
    assert np.isnan(block_bootstrap_ci([0.5], block=3).low)
