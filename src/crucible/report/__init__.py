"""
crucible.report — a self-contained HTML tearsheet for a TradeLog.

Requires the [report] extra (plotly):  pip install "crucible-quant[report]"

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
    tearsheet,
    gauntlet_report,
    verdict_banner,
    verdict_summary,
    pillar_bullets,
    gate_block,
    edge_panels,
    metrics_table,
    report_css,
    title_lockup,
    cumulative_r,
    monthly_r,
)
from crucible.report.scorecards import fullrange_scorecard, holdout_report

__all__ = [
    "tearsheet",
    "gauntlet_report",
    "fullrange_scorecard",
    "holdout_report",
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
]
