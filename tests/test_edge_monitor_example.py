"""Reproducibility guard for the §14 edge-monitor worked example.

The tutorial quotes these exact numbers, so they must not drift silently. A
dependency bump that moves an ARL or flips a verdict should fail here, not
surface as a wrong number in the published page.
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
from examples.edge_monitor import main, synthetic_book, thin  # noqa: E402


@pytest.fixture(scope="module")
def baseline():
    validated = synthetic_book(2000, seed=1, start="2015-01-01")
    return validated, EdgeBaseline.from_log(
        validated, deflated_expectancy=float(validated.r.mean()) * 0.8, n_variants=64)


def test_step1_frozen_baseline(baseline):
    validated, base = baseline
    assert validated.r.mean() == pytest.approx(0.2088, abs=1e-4)
    assert base.expectancy == pytest.approx(0.1671, abs=1e-4)
    assert base.trades_per_year == pytest.approx(150, abs=1)
    assert base.deflated is True
    # the naive alternative, and the 25% flattery the tutorial quotes
    naive = EdgeBaseline.from_log(validated)
    assert naive.deflated is False
    assert naive.expectancy / base.expectancy == pytest.approx(1.25, abs=1e-3)


def test_step2_design_and_its_measured_arls(baseline):
    validated, base = baseline
    d = cusum_design(base)
    assert d.k_r == pytest.approx(0.1253, abs=1e-4)
    assert d.h_std == pytest.approx(46.98, abs=0.05)
    assert d.arl0 == pytest.approx(7_500, rel=1e-3)
    assert d.arl1 == pytest.approx(1_099, abs=5)
    # k is the textbook midpoint (mu_0 + mu_1) / 2
    assert d.k_r == pytest.approx(0.75 * base.expectancy, rel=1e-9)

    in_control = empirical_arl(d, validated.r, baseline_expectancy=base.expectancy,
                               n_sims=300, seed=7)
    assert in_control.mean_run == pytest.approx(7_188, abs=60)
    assert in_control.median_run == pytest.approx(5_587, abs=60)
    assert in_control.inflation == pytest.approx(0.96, abs=0.01)

    halved = empirical_arl(d, validated.r, baseline_expectancy=base.expectancy,
                           shift=0.5, n_sims=300, seed=7)
    assert halved.mean_run == pytest.approx(1_118, abs=20)
    assert halved.median_run == pytest.approx(916, abs=20)
    # the right skew the tutorial calls out: median well under mean
    assert halved.median_run < halved.mean_run * 0.9


def test_step3_three_live_books(baseline):
    _, base = baseline
    healthy = synthetic_book(1300, edge=1.0, seed=2, start="2022-01-01")
    halved = synthetic_book(1300, edge=0.5, seed=3, start="2022-01-01")
    drying = thin(healthy, 3)

    intact = edge_monitor(healthy, base)
    assert intact.label == "HOLDING"
    assert intact.edge_ratio == pytest.approx(1.10, abs=0.01)
    assert intact.cusum_peak / intact.cusum_h == pytest.approx(0.53, abs=0.01)

    decayed = edge_monitor(halved, base)
    assert decayed.label == "DEGRADED"
    assert decayed.alarm_index == 1219
    assert decayed.edge_ratio == pytest.approx(-0.16, abs=0.01)

    thinned = edge_monitor(drying, base)
    assert thinned.label == "SLIPPING"
    assert thinned.n_live == 434
    assert thinned.frequency_ratio == pytest.approx(0.33, abs=0.01)
    # the point of the row: per-trade edge is INTACT, only the rate collapsed
    assert thinned.edge_ratio == pytest.approx(0.88, abs=0.01)
    assert not thinned.cusum_alarm


def test_step4_the_soft_channel_is_noisy_on_a_healthy_book(baseline):
    _, base = baseline
    healthy = synthetic_book(1300, edge=1.0, seed=2, start="2022-01-01")
    roll = rolling_expectancy(healthy, Thresholds().monitor_window).dropna() / base.expectancy

    assert roll.min() == pytest.approx(-0.31, abs=0.01)
    assert roll.max() == pytest.approx(2.40, abs=0.01)
    below = float((roll < Thresholds().monitor_slip_ratio).mean())
    assert below == pytest.approx(0.09, abs=0.01)
    # ...while the calibrated detector stays silent on the same book
    assert not edge_monitor(healthy, base).cusum_alarm


def test_the_example_runs(capsys):
    main()
    out = capsys.readouterr().out
    for expected in ("HOLDING", "DEGRADED", "SLIPPING", "THE FROZEN BASELINE"):
        assert expected in out
