"""Deflating a per-trade expectancy for the size of the search that found it.

`deflated_sharpe` answers "is it real?" with a probability. A monitor cannot anchor to a
probability, so this answers "how much of it should I believe?" with a number in R. The
two share one bar (`_expected_max_sharpe`) and must not drift apart.
"""
import numpy as np
import pytest

from crucible.edge import TradeLog
from crucible.validation import (
    DeflatedExpectancy,
    EdgeBaseline,
    deflated_expectancy,
    deflated_sharpe,
)
from crucible.validation.search_space import SearchSpaceLog


def _trials(mu, n_trials, n_trades, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    return [rng.normal(mu, sigma, n_trades) for _ in range(n_trials)]


def _winner(trials):
    sr = [t.mean() / t.std(ddof=1) for t in trials]
    return trials[int(np.argmax(sr))]


# ── the arithmetic ──────────────────────────────────────────────────────────────────

def test_the_haircut_is_sigma_times_the_bar():
    """deflated = mu - sigma * SR0. The bar is in Sharpe units, so it has to be carried
    back into R before it can be subtracted from an expectancy."""
    t = _trials(0.2, 8, 2000, seed=1)
    d = deflated_expectancy(t[0], t)
    assert d.haircut == pytest.approx(d.sigma * d.sr0_threshold)
    assert d.deflated_expectancy == pytest.approx(d.observed_expectancy - d.haircut)
    assert d.observed_expectancy == pytest.approx(float(np.mean(t[0])))
    assert d.sigma == pytest.approx(float(np.std(t[0], ddof=1)))


def test_it_shares_one_bar_with_the_deflated_sharpe():
    """Same SR0, reached from both directions. Two corrections for one search that
    disagreed about how big the search was would be worse than either alone."""
    t = _trials(0.2, 10, 1500, seed=2)
    w = _winner(t)
    de = deflated_expectancy(w, t)
    ds = deflated_sharpe([x.mean() / x.std(ddof=1) for x in t], returns=w)
    assert de.sr0_threshold == pytest.approx(ds.sr0_threshold)


def test_a_bigger_search_takes_a_bigger_haircut():
    t = _trials(0.2, 6, 3000, seed=3)
    bars = [deflated_expectancy(t[0], t, n_trials=N).haircut
            for N in (6, 50, 500, 5000)]
    assert bars == sorted(bars)
    assert bars[0] < bars[-1]


def test_retained_is_the_surviving_fraction():
    t = _trials(0.3, 5, 4000, seed=4)
    d = deflated_expectancy(t[0], t)
    assert d.retained == pytest.approx(d.deflated_expectancy / d.observed_expectancy)
    assert 0 < d.retained < 1


# ── what it is, and what it is not ──────────────────────────────────────────────────

def test_the_haircut_is_a_bias_correction_not_a_test():
    """The reproducer for the figures quoted in `DeflatedExpectancy.is_positive`.

    Under pure noise the raw mean of the winner is positive essentially always (it was
    selected for being the maximum). Subtracting the EXPECTED maximum removes the average
    selection bias, and the realized maximum sits above its own mean about half the time,
    so the deflated number clears zero roughly as often as not. `deflated_sharpe`, which
    is an actual test, calls none of them significant.

    This is why `is_positive` must not be read as a pass.
    """
    rng = np.random.default_rng(7)
    reps = 400
    out = {}
    for n_trials in (5, 20, 100):
        pos_raw = pos_def = sig = 0
        for _ in range(reps):
            r = rng.normal(0.0, 1.0, (n_trials, 400))
            sr = r.mean(1) / r.std(1, ddof=1)
            win = r[int(np.argmax(sr))]
            d = deflated_expectancy(win, list(r))
            pos_raw += d.observed_expectancy > 0
            pos_def += d.is_positive
            sig += deflated_sharpe(list(sr), returns=win).deflated_sharpe >= 0.95
        out[n_trials] = (pos_raw / reps, pos_def / reps, sig / reps)

    for n_trials, (raw, deflated, significant) in out.items():
        assert raw >= 0.95, f"N={n_trials}: the winner's raw mean should be positive"
        assert 0.35 <= deflated <= 0.65, f"N={n_trials}: expected ~half, got {deflated:.0%}"
        assert significant <= 0.02, f"N={n_trials}: DSR should reject noise, got {significant:.0%}"

    # and the correction still does its job: it strips essentially all of the fake edge
    assert out[100][1] < out[100][0]


def test_it_over_corrects_a_real_edge_which_is_the_safe_direction_for_a_monitor():
    """A winner chosen partly for real signal carries less selection bias than the
    pure-luck maximum being subtracted, so the deflated number sits BELOW the truth."""
    true_mu = 0.20
    t = _trials(true_mu, 40, 2000, seed=11)
    d = deflated_expectancy(_winner(t), t)
    assert d.deflated_expectancy < d.observed_expectancy
    assert d.deflated_expectancy < true_mu * 1.02


# ── the honest N ────────────────────────────────────────────────────────────────────

def test_a_search_space_log_is_accepted_directly():
    log = SearchSpaceLog(scope="trend:arm_x_regime")
    for i in range(60):
        log.record({"variant": i})
    t = _trials(0.2, 5, 2000, seed=5)
    assert (deflated_expectancy(t[0], t, n_trials=log).n_trials
            == deflated_expectancy(t[0], t, n_trials=60).n_trials == 60)


def test_claiming_fewer_trials_than_were_scored_is_refused():
    t = _trials(0.2, 9, 1000, seed=6)
    with pytest.raises(ValueError, match="cannot have tried fewer configs than it scored"):
        deflated_expectancy(t[0], t, n_trials=3)


def test_omitting_the_honest_n_prices_it_at_what_was_supplied():
    t = _trials(0.2, 7, 1000, seed=8)
    assert deflated_expectancy(t[0], t).n_trials == 7


# ── refusals ────────────────────────────────────────────────────────────────────────

def test_one_trial_is_not_a_search():
    """The expected maximum of a single draw is not defined by this approximation, and
    there is no multiple testing to correct for anyway."""
    t = _trials(0.2, 1, 500, seed=9)
    with pytest.raises(ValueError, match=">= 2 scoreable trial logs"):
        deflated_expectancy(t[0], t)


def test_trials_too_thin_to_score_are_dropped_not_counted():
    t = _trials(0.2, 4, 800, seed=10)
    d = deflated_expectancy(t[0], t + [np.array([0.5]), np.array([])])
    assert d.n_scored == 4


def test_a_flat_winner_has_no_sharpe_to_deflate():
    with pytest.raises(ValueError, match="zero dispersion"):
        deflated_expectancy(np.ones(100), _trials(0.2, 4, 500, seed=12))


def test_too_few_trades_to_measure_an_expectancy():
    with pytest.raises(ValueError, match=">= 2 trades"):
        deflated_expectancy([0.5], _trials(0.2, 4, 500, seed=13))


# ── the seam into the monitor ───────────────────────────────────────────────────────

def test_the_baseline_takes_the_result_object_and_marks_itself_deflated():
    t = _trials(0.25, 8, 3000, seed=14)
    w = _winner(t)
    d = deflated_expectancy(w, t)
    b = EdgeBaseline.from_log(TradeLog.from_arrays(w), deflated_expectancy=d)
    assert b.deflated is True
    assert b.expectancy == pytest.approx(d.deflated_expectancy)
    assert b.expectancy < float(np.mean(w))          # strictly below the optimized number


def test_a_float_still_works_and_agrees_with_the_object():
    t = _trials(0.25, 8, 3000, seed=14)
    w = _winner(t)
    d = deflated_expectancy(w, t)
    log = TradeLog.from_arrays(w)
    assert (EdgeBaseline.from_log(log, deflated_expectancy=d.deflated_expectancy)
            == EdgeBaseline.from_log(log, deflated_expectancy=d))


def test_omitting_it_still_falls_back_to_the_optimized_mean():
    t = _trials(0.25, 8, 3000, seed=14)
    b = EdgeBaseline.from_log(TradeLog.from_arrays(t[0]))
    assert b.deflated is False
    assert b.expectancy == pytest.approx(float(np.mean(t[0])))


def test_a_correction_that_leaves_nothing_refuses_to_become_a_baseline():
    """Not a monitor to build with a smaller number: a book that did not survive its
    own search. The refusal names the deflation as a possible cause."""
    rng = np.random.default_rng(15)
    trials = [rng.normal(0.0, 1.0, 300) for _ in range(80)]
    w = _winner(trials)
    d = deflated_expectancy(w, trials, n_trials=100_000)
    assert not d.is_positive
    with pytest.raises(ValueError, match="did not survive its own search"):
        EdgeBaseline.from_log(TradeLog.from_arrays(w), deflated_expectancy=d)


def test_the_deflated_baseline_lowers_the_monitors_bar_rather_than_raising_it():
    """The point of the whole exercise. Anchoring to the optimized mean tells the monitor
    to expect more than the book can deliver, and it alarms on a book performing exactly
    as it truly should."""
    from crucible.validation import cusum_design

    t = _trials(0.25, 30, 3000, seed=16)
    w = _winner(t)
    log = TradeLog.from_arrays(w)
    raw = EdgeBaseline.from_log(log)
    corrected = EdgeBaseline.from_log(log, deflated_expectancy=deflated_expectancy(w, t))
    assert corrected.expectancy < raw.expectancy
    assert cusum_design(corrected).k_r < cusum_design(raw).k_r


def test_the_result_is_a_frozen_readable_record():
    t = _trials(0.2, 6, 1200, seed=17)
    d = deflated_expectancy(t[0], t)
    assert isinstance(d, DeflatedExpectancy)
    text = str(d)
    assert "DEFLATED EXPECTANCY" in text
    assert "NOT a significance test" in text
