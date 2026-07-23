"""
crucible.report — a self-contained HTML tearsheet for a TradeLog.

Requires the [report] extra (plotly):  pip install "crucible[report]"

True to the rest of crucible, these blocks visualize the EDGE, not an account:
the R-multiple distribution, cumulative R, MFE/MAE excursion, and the bootstrap
expectancy distribution behind the HELD/FRAGILE/FAIL verdict. No equity curve, no
capital.

`tearsheet` is the one-call single-book page. The composable blocks
(`verdict_banner`, `edge_panels`, `metrics_table`, `gate_block`, `report_css`)
and the full-page `gauntlet_report` assemble a gauntlet-organized page
(REAL / STRONG / DURABLE / GENERAL) that a host application can extend.
"""
from crucible.report.tearsheet import (
    concurrency_timeline,
    cumulative_r,
    edge_panels,
    edge_ratio_curve,
    equity_drawdown,
    exit_efficiency_dist,
    exit_reason_breakdown,
    gate_block,
    gauntlet_report,
    gross_net_equity,
    holding_vs_r,
    metrics_table,
    monthly_r,
    pillar_bullets,
    report_css,
    segment_forest,
    tearsheet,
    title_lockup,
    verdict_banner,
    verdict_summary,
)

__all__ = [
    "tearsheet",
    "gauntlet_report",
    "verdict_banner",
    "verdict_summary",
    "pillar_bullets",
    "gate_block",
    "edge_panels",
    "metrics_table",
    "report_css",
    "title_lockup",
    "cumulative_r",
    "monthly_r",
    # extra capital-free panels (composable, like edge_panels)
    "equity_drawdown",
    "exit_reason_breakdown",
    "holding_vs_r",
    "exit_efficiency_dist",
    "edge_ratio_curve",
    "gross_net_equity",
    "concurrency_timeline",
    "segment_forest",
]
