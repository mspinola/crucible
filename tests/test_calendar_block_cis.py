"""Calendar-block CIs on a trade log, and their soft companion checks in STRONG."""
import numpy as np
import pandas as pd
import pytest

from crucible.edge import (
    TradeLog,
    bootstrap_metric_cis,
    calendar_block_labels,
    calendar_block_metric_cis,
)
from crucible.validation import Thresholds
from crucible.validation.gauntlet import gate_strong

FAST = Thresholds(n_boot=1_000, n_perm=500, n_random_sims=100)


def _shocked_book(n=800, years=20, shock_sd=0.6, mean=0.35, seed=1):
    """A pooled-looking book: trades scattered over `years`, every trade in a calendar
    year sharing that year's shock."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2000-01-01", periods=years * 261, freq="B")
    dates = days[rng.choice(len(days), n)]
    shock = {y: rng.normal(0, shock_sd) for y in range(2000, 2000 + years + 1)}
    r = rng.normal(mean, 1.5, n) + np.array([shock[d.year] for d in dates])
    return TradeLog(pd.DataFrame({"r": r, "entry_date": dates}))


def test_labels_anchor_on_the_calendar():
    d = ["2020-01-02", "2020-06-30", "2020-07-01", "2021-03-15"]
    h = calendar_block_labels(d, 6)
    assert h[0] == h[1] != h[2] and h[3] == h[2] + 1
    q = calendar_block_labels(d, 3)
    assert len(set(q)) == 4


def test_labels_reject_missing_dates_and_bad_lengths():
    with pytest.raises(ValueError, match="no entry_date"):
        calendar_block_labels(["2020-01-02", None], 6)
    with pytest.raises(ValueError, match="block_months"):
        calendar_block_labels(["2020-01-02"], 0)


def test_point_estimates_match_the_iid_bootstrap():
    tl = _shocked_book()
    blk = calendar_block_metric_cis(tl, n_boot=200)
    iid = bootstrap_metric_cis(tl, n_boot=200)
    for k in ("expectancy", "profit_factor", "win_rate"):
        assert np.isclose(blk[k].point, iid[k].point)


def test_constant_within_block_stays_inside_the_block_values():
    """Every trade in a half-year has the same R: any draw's expectancy is a weighted mix of
    the block values, so the CI lies inside their range; one value everywhere gives zero
    width."""
    dates = list(pd.date_range("2020-01-06", periods=20, freq="W")) + \
        list(pd.date_range("2020-07-06", periods=20, freq="W"))
    tl = TradeLog(pd.DataFrame({"r": [1.0] * 20 + [3.0] * 20, "entry_date": dates}))
    ci = calendar_block_metric_cis(tl, n_boot=500)["expectancy"]
    assert 1.0 <= ci.low <= ci.high <= 3.0
    flat = TradeLog(pd.DataFrame({"r": [0.7] * 40, "entry_date": dates}))
    ci = calendar_block_metric_cis(flat, n_boot=200)["expectancy"]
    assert np.isclose(ci.low, 0.7) and np.isclose(ci.high, 0.7)


def test_a_shared_block_shock_widens_the_interval():
    tl = _shocked_book(shock_sd=0.6)
    blk = calendar_block_metric_cis(tl, n_boot=2_000)["expectancy"]
    iid = bootstrap_metric_cis(tl, n_boot=2_000)["expectancy"]
    assert (blk.high - blk.low) > 1.5 * (iid.high - iid.low)


def test_independent_trades_give_about_the_iid_width():
    tl = _shocked_book(shock_sd=0.0)
    blk = calendar_block_metric_cis(tl, n_boot=2_000)["expectancy"]
    iid = bootstrap_metric_cis(tl, n_boot=2_000)["expectancy"]
    assert 0.7 < (blk.high - blk.low) / (iid.high - iid.low) < 1.4


def test_seeded_and_reproducible():
    tl = _shocked_book()
    a = calendar_block_metric_cis(tl, n_boot=300, seed=7)["profit_factor"]
    b = calendar_block_metric_cis(tl, n_boot=300, seed=7)["profit_factor"]
    assert (a.low, a.high) == (b.low, b.high)


def test_no_losers_gives_an_infinite_profit_factor():
    dates = pd.date_range("2020-01-06", periods=30, freq="W")
    ci = calendar_block_metric_cis(TradeLog(pd.DataFrame({"r": 1.0, "entry_date": dates})),
                                   n_boot=100)["profit_factor"]
    assert ci.point == float("inf") and ci.low == float("inf")


def test_requires_an_entry_date():
    with pytest.raises(ValueError, match="entry_date"):
        calendar_block_metric_cis(TradeLog(pd.DataFrame({"r": [1.0, -1.0]})))


def _checks(g):
    return {c.name: c for c in g.checks}


def test_strong_block_checks_are_soft_and_never_change_the_verdict():
    """A book whose iid floors pass but whose block floors do not: the gate still passes on
    its hard checks, and the block checks come back as soft warnings."""
    tl = _shocked_book(n=2_000, shock_sd=1.0, mean=0.25, seed=3)
    g = gate_strong(tl, thr=FAST)
    c = _checks(g)
    assert c["expectancy_ci_lower"].passed and c["profit_factor_ci_lower"].passed
    for name in ("expectancy_block_ci_lower", "profit_factor_block_ci_lower"):
        assert not c[name].hard
    assert not c["expectancy_block_ci_lower"].passed
    assert not c["profit_factor_block_ci_lower"].passed
    assert c["profit_factor_block_ci_lower"].value < c["profit_factor_ci_lower"].value
    assert g.passed


def test_strong_block_checks_skip_without_dates_and_flag_too_few_blocks():
    g = gate_strong(TradeLog(pd.DataFrame({"r": np.linspace(-1, 3, 60)})), thr=FAST)
    c = _checks(g)
    assert "skipped" in c["expectancy_block_ci_lower"].detail
    assert not c["expectancy_block_ci_lower"].hard

    dates = pd.date_range("2020-01-06", periods=60, freq="D")     # one half-year block
    g = gate_strong(TradeLog(pd.DataFrame({"r": np.linspace(-1, 3, 60), "entry_date": dates})),
                    thr=FAST)
    c = _checks(g)
    assert "not read" in c["profit_factor_block_ci_lower"].detail
    assert not c["profit_factor_block_ci_lower"].passed
