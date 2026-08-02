"""The decay panel. Behind the [report] extra, like every other plotly block."""
import numpy as np
import pytest

from crucible.edge import TradeLog
from crucible.validation import EdgeBaseline, Thresholds, cusum_design, cusum_path

pytest.importorskip("plotly")

from crucible.report import monitor_panel  # noqa: E402


def _base():
    return EdgeBaseline(expectancy=0.2, sigma=1.0, n_trades=2000, deflated=True)


def _log(mu, n=600, seed=0):
    return TradeLog.from_arrays(np.random.default_rng(seed).normal(mu, 1.0, n))


# ── cusum_path, the series the panel plots ──────────────────────────────────────────

def test_cusum_path_is_one_value_per_trade_and_never_negative():
    p = cusum_path(_log(0.2), _base())
    assert len(p) == 600
    assert (p >= 0).all()


def test_cusum_path_agrees_with_the_verdict_it_reduces_to():
    from crucible.validation import edge_monitor

    base, live = _base(), _log(0.1, n=3000, seed=5)
    th = Thresholds(monitor_arl0_trades=500)
    p = cusum_path(live, base, thresholds=th)
    v = edge_monitor(live, base, thresholds=th)
    assert p.iloc[-1] == pytest.approx(v.cusum_now)
    assert p.max() == pytest.approx(v.cusum_peak)
    # the verdict's alarm index is the first crossing of the plotted boundary
    d = cusum_design(base, thresholds=th)
    assert int(np.flatnonzero(p.to_numpy() > d.h_std)[0]) + 1 == v.alarm_index


def test_cusum_path_climbs_on_a_decayed_book_and_not_on_a_healthy_one():
    base = _base()
    healthy = cusum_path(_log(0.2, n=1500, seed=1), base).max()
    decayed = cusum_path(_log(0.05, n=1500, seed=1), base).max()
    assert decayed > healthy * 3


def test_cusum_path_cannot_rebuild_a_baseline():
    import inspect

    assert set(inspect.signature(cusum_path).parameters) == {
        "trades", "baseline", "design", "thresholds"}


# ── the panel ───────────────────────────────────────────────────────────────────────

def test_panel_renders_both_rows():
    html = monitor_panel(_log(0.2), _base())
    assert "plotly" in html.lower()
    assert "CUSUM" in html and "baseline" in html


def test_panel_marks_the_alarm_when_one_fired():
    live = _log(0.05, n=2000, seed=3)
    html = monitor_panel(live, _base(), thresholds=Thresholds(monitor_arl0_trades=500))
    assert "ALARM at trade" in html


def test_panel_omits_the_alarm_marker_on_a_healthy_book():
    html = monitor_panel(_log(0.2, n=800, seed=2), _base())
    assert "ALARM at trade" not in html


def test_panel_degrades_without_a_full_rolling_window():
    """Fewer trades than the window: the CUSUM row still renders, since a sequential
    detector is meaningful from trade one, and only the trailing read is skipped."""
    html = monitor_panel(_log(0.2, n=40), _base(), thresholds=Thresholds(monitor_window=200))
    assert "CUSUM" in html
    assert "trailing 200-trade E" not in html


def test_panel_returns_empty_string_for_an_empty_log():
    assert monitor_panel(TradeLog.from_arrays([]), _base()) == ""


def test_alarm_boundary_stays_in_frame_on_a_healthy_book():
    """Autoscaling the CUSUM row to the data alone would push h out of view on exactly
    the books where the useful reading is how much room is left. On the default design
    a healthy book peaks around 25 sigma against an h of 47."""
    base = _base()
    live = _log(0.3, n=1300, seed=2)                    # comfortably above baseline
    h = cusum_design(base).h_std
    assert cusum_path(live, base).max() < h * 0.5       # the condition that exposes it
    html = monitor_panel(live, base)
    # plotly writes the pinned axis range into the figure JSON
    assert f'"range":[0,{h * 1.12}]'.replace(" ", "") in html.replace(" ", "")
