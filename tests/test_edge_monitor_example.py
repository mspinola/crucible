"""Reproducibility guard for the §14 edge-monitor worked example.

The tutorial quotes these exact numbers, so they must not drift silently. A
dependency bump that moves an ARL or flips a verdict should fail here, not
surface as a wrong number in the published page.

The baseline comes from `examples.edge_monitor.promoted_book`, NOT from a copy of
its setup. An earlier version of this file rebuilt the baseline itself, so when the
example switched from a hardcoded 20% haircut to a real `deflated_expectancy` every
assertion here still passed while every figure in the example changed. A guard that
reconstructs what it guards is not a guard.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crucible.validation import (  # noqa: E402
    EdgeBaseline,
    Thresholds,
    cusum_design,
    edge_monitor,
    empirical_arl,
    rolling_expectancy,
)
from examples.edge_monitor import (  # noqa: E402
    main,
    promoted_book,
    synthetic_book,
    thin,
)


@pytest.fixture(scope="module")
def promoted():
    return promoted_book()


def test_step1_the_search_and_the_deflated_baseline(promoted):
    validated, corrected, base = promoted
    assert validated.r.mean() == pytest.approx(0.2751, abs=1e-4)

    # the haircut is sigma x SR0, priced against the honest N of 64
    assert corrected.n_trials == 64
    assert corrected.sr0_threshold == pytest.approx(0.0503, abs=1e-4)
    assert corrected.sigma == pytest.approx(1.4386, abs=1e-4)
    assert corrected.haircut == pytest.approx(0.0723, abs=1e-4)
    assert corrected.retained == pytest.approx(0.737, abs=1e-3)
    assert corrected.is_positive

    assert base.expectancy == pytest.approx(0.2027, abs=1e-4)
    assert base.trades_per_year == pytest.approx(150, abs=1)
    assert base.deflated is True

    # the naive alternative, and the flattery the tutorial quotes
    naive = EdgeBaseline.from_log(validated)
    assert naive.deflated is False
    assert naive.expectancy / base.expectancy == pytest.approx(1.357, abs=1e-3)


def test_step2_design_and_its_measured_arls(promoted):
    validated, _, base = promoted
    d = cusum_design(base)
    assert d.k_r == pytest.approx(0.1520, abs=1e-4)
    assert d.h_std == pytest.approx(35.09, abs=0.05)
    # the budget is calendar time: 25 years at this book's 150 trades/yr
    assert d.arl0_basis == "years"
    assert d.arl0 == pytest.approx(3_752, abs=5)
    assert d.arl0 / base.trades_per_year == pytest.approx(25.0, abs=0.1)
    assert d.arl1 == pytest.approx(658, abs=5)
    assert d.arl1 / base.trades_per_year == pytest.approx(4.4, abs=0.1)
    # k is the textbook midpoint (mu_0 + mu_1) / 2
    assert d.k_r == pytest.approx(0.75 * base.expectancy, rel=1e-9)

    in_control = empirical_arl(d, validated.r, baseline_expectancy=base.expectancy,
                               n_sims=300, seed=7)
    assert in_control.mean_run == pytest.approx(4_087, abs=60)
    assert in_control.median_run == pytest.approx(3_244, abs=60)
    assert in_control.inflation == pytest.approx(1.09, abs=0.02)

    halved = empirical_arl(d, validated.r, baseline_expectancy=base.expectancy,
                           shift=0.5, n_sims=300, seed=7)
    assert halved.mean_run == pytest.approx(632, abs=20)
    assert halved.median_run == pytest.approx(500, abs=20)
    # the right skew the tutorial calls out: median well under mean
    assert halved.median_run < halved.mean_run * 0.9


def test_step3_three_live_books(promoted):
    _, _, base = promoted
    healthy = synthetic_book(1300, edge=1.0, seed=2, start="2022-01-01")
    halved = synthetic_book(1300, edge=0.5, seed=3, start="2022-01-01")
    drying = thin(healthy, 3)

    intact = edge_monitor(healthy, base)
    assert intact.label == "HOLDING"
    assert intact.edge_ratio == pytest.approx(0.91, abs=0.01)
    assert intact.cusum_peak / intact.cusum_h == pytest.approx(0.82, abs=0.01)

    decayed = edge_monitor(halved, base)
    assert decayed.label == "DEGRADED"
    assert decayed.alarm_index == 276
    assert decayed.edge_ratio == pytest.approx(-0.13, abs=0.01)

    thinned = edge_monitor(drying, base)
    assert thinned.label == "SLIPPING"
    assert thinned.n_live == 434
    assert thinned.frequency_ratio == pytest.approx(0.33, abs=0.01)
    # the point of the row: per-trade edge is INTACT, only the rate collapsed
    assert thinned.edge_ratio == pytest.approx(0.72, abs=0.01)
    assert not thinned.cusum_alarm


def test_step4_the_soft_channel_is_noisy_on_a_healthy_book(promoted):
    _, _, base = promoted
    healthy = synthetic_book(1300, edge=1.0, seed=2, start="2022-01-01")
    roll = rolling_expectancy(healthy, Thresholds().monitor_window).dropna() / base.expectancy

    assert roll.min() == pytest.approx(-0.25, abs=0.01)
    assert roll.max() == pytest.approx(1.97, abs=0.01)
    below = float((roll < Thresholds().monitor_slip_ratio).mean())
    assert below == pytest.approx(0.11, abs=0.01)
    # ...while the calibrated detector stays silent on the same book
    assert not edge_monitor(healthy, base).cusum_alarm


def test_the_deflation_is_what_keeps_the_healthy_book_holding(promoted):
    """The payoff, stated as a test. Anchored to the optimized mean, a book whose true
    edge never moved reads as decayed; anchored to the corrected one, it reads HOLDING."""
    validated, _, base = promoted
    healthy = synthetic_book(1300, edge=1.0, seed=2, start="2022-01-01")
    naive = EdgeBaseline.from_log(validated)

    assert edge_monitor(healthy, base).label == "HOLDING"
    assert edge_monitor(healthy, naive).label == "DEGRADED"


def test_the_example_runs(capsys):
    main()
    out = capsys.readouterr().out
    for expected in ("HOLDING", "DEGRADED", "SLIPPING", "THE FROZEN BASELINE",
                     "search haircut"):
        assert expected in out
