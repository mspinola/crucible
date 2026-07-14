import numpy as np

from crucible.edge import (
    TradeLog, edge_report, expectancy, profit_factor, payoff_ratio,
    win_rate, sqn,
)


def test_expectancy_hand_computed():
    # 2 wins (+2R, +2R), 2 losses (-1R, -1R): E = 0.5*2 - 0.5*1 = +0.5
    r = [2.0, 2.0, -1.0, -1.0]
    assert expectancy(r) == 0.5
    assert win_rate(r) == 0.5
    assert payoff_ratio(r) == 2.0
    assert profit_factor(r) == 2.0  # 4 / 2


def test_profit_factor_no_losers_is_inf():
    assert profit_factor([1.0, 2.0]) == float("inf")


def test_sqn_zero_when_no_dispersion():
    assert sqn([1.0, 1.0, 1.0]) == 0.0


def test_edge_report_uses_available_columns():
    tl = TradeLog.from_arrays(
        r=[2.0, -1.0, 2.0, -1.0],
        mfe=[2.5, 0.3, 3.0, 0.2],
        mae=[-0.4, -1.0, -0.5, -1.0],
        bars_held=[10, 3, 12, 2],
    )
    rep = edge_report(tl)
    assert rep.n == 4
    assert rep.expectancy == 0.5
    assert rep.excursion_ratio is not None and rep.excursion_ratio > 1
    assert rep.time_asymmetry is not None and rep.time_asymmetry > 1
    assert "EDGE REPORT" in str(rep)


def test_empty_is_safe():
    assert expectancy([]) == 0.0
