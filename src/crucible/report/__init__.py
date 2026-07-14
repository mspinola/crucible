"""
crucible.report — a self-contained HTML tearsheet for a TradeLog.

Requires the [report] extra (plotly):  pip install "crucible-quant[report]"

True to the rest of crucible, the tearsheet visualizes the EDGE, not an account:
the R-multiple distribution, cumulative R, MFE/MAE excursion, and the bootstrap
expectancy distribution behind the HELD/FRAGILE/FAIL verdict. No equity curve, no
capital.
"""
from crucible.report.tearsheet import tearsheet, cumulative_r

__all__ = ["tearsheet", "cumulative_r"]
