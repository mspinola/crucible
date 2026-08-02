"""The edge monitor: has a promoted edge decayed?

The load-bearing tests here are the calibration ones. A monitor whose stated
false-alarm rate is fiction is worse than no monitor, because it launders a guess
as a number.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from crucible.edge import TradeLog
from crucible.validation import (
    CusumDesign,
    EdgeBaseline,
    Thresholds,
    cusum_design,
    edge_monitor,
    empirical_arl,
    rolling_expectancy,
)
from crucible.validation.monitor import DEGRADED, HOLDING, SLIPPING


def _log(r, start="2020-01-01", freq="3D"):
    dates = pd.date_range(start, periods=len(r), freq=freq)
    return TradeLog.from_arrays(r, entry_date=dates, exit_date=dates)


def _typical(n, mu_scale=1.0, seed=0):
    """A realistic trade shape: 43% win rate, small losses, larger wins."""
    rng = np.random.default_rng(seed)
    win = rng.random(n) < 0.43
    r = np.where(win, rng.normal(1.8, 0.6, n), rng.normal(-0.95, 0.25, n))
    return r * mu_scale if mu_scale == 1.0 else r - (1 - mu_scale) * r.mean()


# ── EdgeBaseline ────────────────────────────────────────────────────────────────────

def test_baseline_refuses_a_non_positive_edge():
    with pytest.raises(ValueError, match="must be positive"):
        EdgeBaseline(expectancy=0.0, sigma=1.0, n_trades=100)
    with pytest.raises(ValueError, match="must be positive"):
        EdgeBaseline(expectancy=-0.1, sigma=1.0, n_trades=100)


def test_baseline_refuses_zero_dispersion():
    with pytest.raises(ValueError, match="sigma must be positive"):
        EdgeBaseline(expectancy=0.1, sigma=0.0, n_trades=100)


def test_from_log_records_whether_the_baseline_was_deflated():
    log = _log(_typical(500))
    raw = EdgeBaseline.from_log(log)
    assert raw.deflated is False
    assert raw.expectancy == pytest.approx(log.r.mean())

    corrected = EdgeBaseline.from_log(log, deflated_expectancy=0.05, n_variants=64)
    assert corrected.deflated is True
    assert corrected.expectancy == 0.05
    assert corrected.n_variants == 64


def test_from_log_derives_the_firing_rate_from_dates():
    log = _log(_typical(200), freq="D")          # one trade a day
    base = EdgeBaseline.from_log(log)
    assert base.trades_per_year == pytest.approx(365.25, rel=0.02)


def test_from_log_leaves_the_firing_rate_off_without_dates():
    base = EdgeBaseline.from_log(TradeLog.from_arrays(_typical(200)))
    assert base.trades_per_year is None


# ── CUSUM design ────────────────────────────────────────────────────────────────────

def test_reference_value_is_the_textbook_midpoint():
    """k = (mu_0 + mu_1)/2. For a halving that is 0.75 * mu_0, the same relationship
    a known-good third-party implementation publishes (k=0.101 against mu_0=0.134)."""
    base = EdgeBaseline(expectancy=0.134, sigma=1.0, n_trades=2603)
    d = cusum_design(base)
    assert d.k_r == pytest.approx(0.75 * 0.134)
    assert d.k_r == pytest.approx(0.101, abs=0.001)


@pytest.mark.parametrize("target", [100, 300, 1_000, 7_500])
def test_solver_hits_the_requested_false_alarm_budget(target):
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=500)
    d = cusum_design(base, thresholds=Thresholds(monitor_arl0_trades=target))
    assert d.arl0 == pytest.approx(target, rel=1e-3)
    assert d.arl1 < d.arl0          # detecting the design shift beats a false alarm


def test_a_stricter_false_alarm_budget_costs_detection_latency():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=500)
    fast = cusum_design(base, thresholds=Thresholds(monitor_arl0_trades=200))
    slow = cusum_design(base, thresholds=Thresholds(monitor_arl0_trades=5_000))
    assert slow.h_std > fast.h_std
    assert slow.arl1 > fast.arl1    # the trade-off is stated, not hidden


def test_a_barely_detectable_shift_states_its_cost_instead_of_hiding_it():
    """An edge tiny relative to per-trade noise still yields a valid design. It is not
    refused, because "useless" is the caller's judgment, but the price is on the face of
    the object: a latency no live book will ever reach."""
    tiny = EdgeBaseline(expectancy=1e-6, sigma=10.0, n_trades=500)
    d = cusum_design(tiny, thresholds=Thresholds(monitor_arl0_trades=10**6))
    assert d.arl0 == pytest.approx(10**6, rel=1e-3)
    assert d.arl1 > 100_000                      # visibly hopeless, and visibly so


def test_an_unreachable_false_alarm_budget_raises():
    tiny = EdgeBaseline(expectancy=1e-6, sigma=10.0, n_trades=500)
    with pytest.raises(ValueError, match="too small"):
        cusum_design(tiny, thresholds=Thresholds(monitor_arl0_trades=10**14))


# ── calibration: the claims the monitor makes about itself ──────────────────────────

def test_nominal_arl0_survives_contact_with_gaussian_data():
    """Validates the whole chain (Siegmund + bisection + the statistic) where the
    theory is exactly right, so a failure here is an implementation bug."""
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=1000)
    d = cusum_design(base, thresholds=Thresholds(monitor_arl0_trades=300))
    g = np.random.default_rng(0).normal(0.2, 1.0, 20_000)
    e = empirical_arl(d, g, baseline_expectancy=0.2, n_sims=400, seed=1)
    assert e.inflation == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("win_rate,payoff", [(0.43, 1.8), (0.10, 12.0)])
def test_gaussian_calibration_holds_on_skewed_trade_returns(win_rate, payoff):
    """The design assumes normal increments and trade R is not normal. It survives
    anyway: the boundary is many sigma out, so the CUSUM aggregates enough increments
    for the CLT to carry it. Documented as a measured bound, not an assumption."""
    rng = np.random.default_rng(7)
    n = 40_000
    win = rng.random(n) < win_rate
    r = np.where(win, rng.normal(payoff, payoff / 3, n), rng.normal(-1.0, 0.2, n))
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    assert mu > 0

    d = cusum_design(EdgeBaseline(expectancy=mu, sigma=sd, n_trades=n),
                     thresholds=Thresholds(monitor_arl0_trades=300))
    e = empirical_arl(d, r, baseline_expectancy=mu, n_sims=600, seed=3)
    assert 0.8 < e.inflation < 1.35


def test_run_length_is_right_skewed_so_mean_and_median_differ():
    """Why both are reported: quoting a median against someone else's mean (or the
    reverse) misstates detection latency by roughly a third."""
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=1000)
    d = cusum_design(base, thresholds=Thresholds(monitor_arl0_trades=300))
    g = np.random.default_rng(0).normal(0.2, 1.0, 20_000)
    e = empirical_arl(d, g, baseline_expectancy=0.2, n_sims=600, seed=2)
    assert e.median_run < e.mean_run * 0.85


def test_empirical_arl_measures_detection_latency_at_the_design_shift():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=1000)
    d = cusum_design(base, thresholds=Thresholds(monitor_arl0_trades=300))
    g = np.random.default_rng(0).normal(0.2, 1.0, 20_000)
    decayed = empirical_arl(d, g, baseline_expectancy=0.2, shift=0.5,
                            n_sims=400, seed=4)
    assert decayed.shift == 0.5
    assert decayed.mean_run < d.arl0        # it notices decay faster than it false-alarms
    assert decayed.inflation == pytest.approx(1.0, abs=0.25)


def test_empirical_arl_is_deterministic_given_a_seed():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=1000)
    d = cusum_design(base, thresholds=Thresholds(monitor_arl0_trades=200))
    g = np.random.default_rng(0).normal(0.2, 1.0, 5_000)
    a = empirical_arl(d, g, baseline_expectancy=0.2, n_sims=100, seed=11)
    b = empirical_arl(d, g, baseline_expectancy=0.2, n_sims=100, seed=11)
    assert a == b


# ── the monitor ─────────────────────────────────────────────────────────────────────

def test_a_stable_edge_holds():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)
    live = _log(np.random.default_rng(5).normal(0.2, 1.0, 400))
    v = edge_monitor(live, base)
    assert v.label == HOLDING
    assert not v.cusum_alarm


def test_a_halved_edge_is_detected():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)
    live = _log(np.random.default_rng(5).normal(0.1, 1.0, 3_000))   # exactly half
    v = edge_monitor(live, base, thresholds=Thresholds(monitor_arl0_trades=500))
    assert v.label == DEGRADED
    assert v.cusum_alarm and v.alarm_index is not None


def test_the_false_alarm_rate_is_roughly_what_was_advertised():
    """Forty independent stable books against a 500-trade ARL0 budget, 300 trades each.
    Expected alarm rate is about 1 - exp(-300/500) = 45%; the point is that it lands in
    the neighbourhood rather than firing on everything or nothing."""
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)
    th = Thresholds(monitor_arl0_trades=500)
    fired = sum(
        edge_monitor(_log(np.random.default_rng(s).normal(0.2, 1.0, 300)),
                     base, thresholds=th).cusum_alarm
        for s in range(40)
    )
    assert 0.20 <= fired / 40 <= 0.70


# ── the split that makes the design honest ──────────────────────────────────────────

def test_only_the_calibrated_detector_can_say_degraded():
    """The rolling ratio has no stated false-alarm rate, so the worst it may say is
    SLIPPING. A boundary the CUSUM cannot reach isolates the label logic."""
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)
    unreachable = CusumDesign(k_r=0.15, k_std=0.05, h_std=1e9, h_r=1e9, sigma=1.0,
                              detect_shift=0.5, arl0=1e12, arl1=1e6)
    live = _log(np.full(300, 0.02))            # 10% of baseline: unmistakably decayed
    v = edge_monitor(live, base, design=unreachable,
                     thresholds=Thresholds(monitor_window=100))
    assert v.edge_ratio < 0.5
    assert not v.cusum_alarm
    assert v.label == SLIPPING                 # NOT degraded


def test_a_frequency_collapse_is_caught_with_expectancy_intact():
    """The failure mode an expectancy-only monitor misses: per-trade edge unchanged,
    the signal just stops firing, and annual R falls with the opportunity set."""
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000,
                        trades_per_year=120.0, deflated=True)
    live = _log(np.random.default_rng(5).normal(0.2, 1.0, 60), freq="15D")  # ~24/yr
    v = edge_monitor(live, base)
    assert v.frequency_ratio < 0.6
    assert not v.cusum_alarm
    assert v.label == SLIPPING
    assert any("fires at" in r for r in v.reasons)


def test_edge_monitor_cannot_rebuild_a_baseline():
    """Trap 2. A monitor that can recompute its own reference re-fits onto the drifted
    reality and can never fire, which looks entirely correct in review."""
    params = set(inspect.signature(edge_monitor).parameters)
    assert params == {"trades", "baseline", "design", "thresholds"}


def test_an_undeflated_baseline_is_allowed_but_never_silent():
    """Trap 3. Anchoring to the optimized in-sample number is a legitimate choice in a
    hurry; hiding that you did it is not."""
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000)   # deflated defaults False
    v = edge_monitor(_log(np.random.default_rng(5).normal(0.2, 1.0, 300)), base)
    assert v.deflated is False
    assert any("NOT search-corrected" in r for r in v.reasons)
    assert "NOT search-corrected" in str(v)


# ── edges and plumbing ──────────────────────────────────────────────────────────────

def test_no_live_trades_yields_a_verdict_rather_than_an_exception():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)
    v = edge_monitor(TradeLog.from_arrays([]), base)
    assert v.label == HOLDING and v.n_live == 0
    assert "nothing to compare" in v.reasons[0]


def test_the_rolling_channel_is_skipped_until_its_window_fills():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)
    v = edge_monitor(_log(np.random.default_rng(1).normal(0.2, 1.0, 50)), base,
                     thresholds=Thresholds(monitor_window=200))
    assert v.recent_expectancy is None and v.edge_ratio is None
    assert any("needs a full window" in r for r in v.reasons)


def test_rolling_expectancy_tracks_a_step_change():
    r = np.concatenate([np.full(50, 0.4), np.full(50, -0.4)])
    s = rolling_expectancy(_log(r), window=10)
    assert s.isna().sum() == 9
    assert s.iloc[49] == pytest.approx(0.4)
    assert s.iloc[-1] == pytest.approx(-0.4)


def test_rolling_expectancy_refuses_a_degenerate_window():
    with pytest.raises(ValueError, match="at least 2"):
        rolling_expectancy(_log(_typical(100)), window=1)


def test_verdict_renders_without_optional_channels():
    base = EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)
    v = edge_monitor(TradeLog.from_arrays(np.random.default_rng(2).normal(0.2, 1, 40)),
                     base)
    out = str(v)
    assert "EDGE MONITOR" in out and "n/a" in out
