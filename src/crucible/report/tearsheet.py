"""A capital-free HTML tearsheet: the edge, its shape, and its verdict."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from crucible.edge.trade_log import TradeLog
from crucible.edge.metrics import edge_report, expectancy
from crucible.edge.stats import reality_check

_VERDICT_COLOR = {"HELD": "#1a7f37", "FRAGILE": "#9a6700", "FAIL": "#b42318"}


def _plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'crucible.report needs plotly — install the extra:\n'
            '    pip install "crucible-quant[report]"'
        ) from e


def cumulative_r(trades: TradeLog) -> pd.Series:
    """Cumulative sum of R over trades ordered by exit date (entry date, or the
    given order, as fallbacks). Capital-free — this is summed R, not an equity
    curve."""
    f = trades.frame
    order = "exit_date" if "exit_date" in f.columns else (
        "entry_date" if "entry_date" in f.columns else None)
    g = f.sort_values(order) if order else f
    x = pd.to_datetime(g[order]) if order else pd.RangeIndex(len(g))
    return pd.Series(np.cumsum(g["r"].to_numpy(dtype=float)), index=x)


def _metrics_table(trades: TradeLog) -> str:
    rep = edge_report(trades).to_dict()
    rows = [
        ("Trades", f"{rep['n']}"),
        ("Win rate", f"{rep['win_rate'] * 100:.1f}%"),
        ("Expectancy", f"{rep['expectancy']:+.3f} R"),
        ("Profit factor", f"{rep['profit_factor']:.2f}"),
        ("Payoff ratio", f"{rep['payoff_ratio']:.2f}"),
        ("SQN-100", f"{rep['sqn']:.2f}"),
    ]
    if rep.get("excursion_ratio") is not None:
        rows.append(("Excursion ratio", f"{rep['excursion_ratio']:.2f}"))
    if rep.get("exit_efficiency") is not None:
        rows.append(("Exit efficiency", f"{rep['exit_efficiency'] * 100:.1f}%"))
    cells = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table class='metrics'>{cells}</table>"


def tearsheet(trades: TradeLog, path: str = "tearsheet.html", *,
              title: str = "crucible tearsheet", subtitle: Optional[str] = None,
              n_boot: int = 10_000, seed: int = 0,
              include_plotlyjs: bool = True) -> str:
    """Write a self-contained HTML tearsheet for `trades` and return its path.

    Panels: R-multiple distribution, cumulative R, MFE-vs-MAE excursion (when
    present), and the bootstrap expectancy distribution with the CI and verdict.
    `include_plotlyjs=True` inlines plotly.js so the file renders offline."""
    go, make_subplots = _plotly()
    r = trades.r
    v = reality_check(trades, n_boot=n_boot, seed=seed)

    # Bootstrap expectancy draws for the distribution panel.
    if len(r):
        rng = np.random.default_rng(seed)
        boot = np.array([expectancy(rng.choice(r, size=len(r), replace=True))
                         for _ in range(min(n_boot, 5000))])
    else:
        boot = np.array([])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("R-multiple distribution", "Cumulative R",
                        "Excursion: MFE vs MAE (R)", "Bootstrap expectancy"),
    )
    # 1) R histogram
    fig.add_trace(go.Histogram(x=r, nbinsx=40, marker_color="#4c78a8",
                               name="R"), row=1, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#888", row=1, col=1)

    # 2) cumulative R
    cr = cumulative_r(trades)
    fig.add_trace(go.Scatter(x=list(cr.index), y=cr.values, mode="lines",
                             line_color="#1a7f37", name="cum R"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#888", row=1, col=2)

    # 3) MFE vs MAE excursion, colored by win/loss
    mfe, mae = trades.col("mfe"), trades.col("mae")
    if mfe is not None and mae is not None:
        win = r > 0
        fig.add_trace(go.Scatter(x=mae[win], y=mfe[win], mode="markers",
                                 marker=dict(color="#1a7f37", size=6, opacity=0.6),
                                 name="win"), row=2, col=1)
        fig.add_trace(go.Scatter(x=mae[~win], y=mfe[~win], mode="markers",
                                 marker=dict(color="#b42318", size=6, opacity=0.6),
                                 name="loss"), row=2, col=1)

    # 4) bootstrap expectancy distribution + CI + point
    if len(boot):
        fig.add_trace(go.Histogram(x=boot, nbinsx=40, marker_color="#b0b0b0",
                                   name="boot E"), row=2, col=2)
        for xval, dash, color in ((0, "dot", "#888"),
                                  (v.ci.low, "dash", "#9a6700"),
                                  (v.ci.high, "dash", "#9a6700"),
                                  (v.point, "solid", "#1a7f37")):
            fig.add_vline(x=xval, line_dash=dash, line_color=color, row=2, col=2)

    fig.update_layout(height=760, showlegend=False, bargap=0.05,
                      margin=dict(t=48, l=48, r=24, b=40),
                      template="plotly_white")
    fig_html = fig.to_html(full_html=False,
                           include_plotlyjs=(True if include_plotlyjs else "cdn"))

    color = _VERDICT_COLOR.get(v.label, "#888")
    sub = f"<div class='sub'>{subtitle}</div>" if subtitle else ""
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
          color: #1a1a1a; background: #fff; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px; }}
  h1 {{ margin: 0 0 2px; font-size: 22px; }}
  .sub {{ color: #666; margin-bottom: 16px; }}
  .verdict {{ display: inline-block; padding: 8px 14px; border-radius: 8px;
              color: #fff; font-weight: 600; background: {color}; margin: 8px 0 4px; }}
  .verdict small {{ font-weight: 400; opacity: .9; }}
  .cols {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }}
  table.metrics {{ border-collapse: collapse; margin: 12px 0; }}
  table.metrics th {{ text-align: left; color: #555; font-weight: 500;
                      padding: 3px 16px 3px 0; }}
  table.metrics td {{ text-align: right; font-variant-numeric: tabular-nums;
                      font-weight: 600; }}
  .foot {{ color: #999; font-size: 12px; margin-top: 20px; }}
</style></head><body><div class="wrap">
  <h1>{title}</h1>{sub}
  <div class="verdict">{v.label} &nbsp;<small>{v.metric} {v.point:+.3f} R &nbsp;
    CI [{v.ci.low:+.3f}, {v.ci.high:+.3f}] &nbsp; p(edge&gt;0)={1 - v.p_value:.3f}</small></div>
  <div class="cols">{_metrics_table(trades)}</div>
  {fig_html}
  <div class="foot">crucible — capital-free edge evaluation. Returns in R-multiples;
    no capital, position sizing, or equity curve.</div>
</div></body></html>"""

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
