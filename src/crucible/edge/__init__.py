"""
crucible.edge — the raw mathematical edge of a signal, capital-free.

Trade log in, edge verdict out. No account, no position sizing, no equity curve.
If you want a $ equity curve, hand the :class:`TradeLog` to quantstats — this
package stops at the edge.
"""
from crucible.edge.metrics import (
    EdgeReport,
    e_ratio,
    edge_report,
    excursion_ratio,
    exit_efficiency,
    expectancy,
    payoff_ratio,
    profit_factor,
    sqn,
    time_asymmetry,
    win_rate,
)
from crucible.edge.simulator import barrier_trades, random_entries
from crucible.edge.stats import (
    CI,
    Verdict,
    block_bootstrap_ci,
    block_bootstrap_pvalue,
    bootstrap_ci,
    bootstrap_metric_cis,
    detrended_timing_null,
    p_value_positive,
    random_entry_null,
    reality_check,
)
from crucible.edge.trade_log import TradeLog

__all__ = [
    "TradeLog",
    "edge_report", "EdgeReport",
    "expectancy", "profit_factor", "payoff_ratio", "win_rate", "sqn",
    "excursion_ratio", "e_ratio", "time_asymmetry", "exit_efficiency",
    "bootstrap_ci", "bootstrap_metric_cis", "p_value_positive", "reality_check",
    "random_entry_null", "detrended_timing_null",
    "block_bootstrap_pvalue", "block_bootstrap_ci", "CI", "Verdict",
    "barrier_trades", "random_entries",
]
