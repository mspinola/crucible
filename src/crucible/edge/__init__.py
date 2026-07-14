"""
crucible.edge — the raw mathematical edge of a signal, capital-free.

Trade log in, edge verdict out. No account, no position sizing, no equity curve.
If you want a $ equity curve, hand the :class:`TradeLog` to quantstats — this
package stops at the edge.
"""
from crucible.edge.trade_log import TradeLog
from crucible.edge.metrics import (
    edge_report, EdgeReport,
    expectancy, profit_factor, payoff_ratio, win_rate, sqn,
    excursion_ratio, e_ratio, time_asymmetry, exit_efficiency,
)
from crucible.edge.stats import (
    bootstrap_ci, p_value_positive, reality_check, random_entry_null,
    CI, Verdict,
)
from crucible.edge.simulator import barrier_trades, random_entries

__all__ = [
    "TradeLog",
    "edge_report", "EdgeReport",
    "expectancy", "profit_factor", "payoff_ratio", "win_rate", "sqn",
    "excursion_ratio", "e_ratio", "time_asymmetry", "exit_efficiency",
    "bootstrap_ci", "p_value_positive", "reality_check", "random_entry_null",
    "CI", "Verdict",
    "barrier_trades", "random_entries",
]
