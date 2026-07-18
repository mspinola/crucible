"""Purpose-built scorecards for two run modes, distinct from the gauntlet report.

`crucible.report.tearsheet` gives you the reality-check tearsheet and the
gauntlet-organized page. Those two deliberately share chrome (logo, verdict pill,
metric row), which makes a plain reality-check tearsheet read like the gauntlet
report at a glance. These two pages are styled apart on purpose so each *run mode*
looks like itself:

* `fullrange_scorecard` — the whole-history reality-check read, led by big stat
  tiles (verdict, total R, R/year) rather than the gauntlet's pillar chips.
* `holdout_report` — the early-train / late-confirm split as two side-by-side
  cards (TRAIN vs the untouched TEST) over a cumulative-R chart split at the
  boundary.

Both are capital-free: R-multiples only, no equity curve.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from crucible.edge.trade_log import TradeLog
from crucible.edge.metrics import expectancy
from crucible.edge.stats import reality_check
from crucible.report.tearsheet import (
    _FOOT, _page, _plotly, _VERDICT_COLOR,
    title_lockup, metrics_table, edge_panels, cumulative_r,
)


def _years_span(trades: TradeLog) -> Optional[float]:
    """Calendar span of the log in years, from exit dates (entry as a fallback),
    or ``None`` when the log carries no usable dates."""
    f = trades.frame
    for col in ("exit_date", "entry_date"):
        if col in f.columns:
            d = pd.to_datetime(f[col])
            span = (d.max() - d.min()).days / 365.25
            return span if span > 0 else None
    return None


def _tile(value: str, label: str, *, bg: Optional[str] = None) -> str:
    cls = "cr-tile cr-verdict-tile" if bg else "cr-tile"
    style = f" style='background:{bg}'" if bg else ""
    return f"<div class='{cls}'{style}><span class='v'>{value}</span><span class='k'>{label}</span></div>"


def fullrange_scorecard(trades: TradeLog, path: Optional[str] = None, *,
                        title: str = "Full-range scorecard", subtitle: Optional[str] = None,
                        n_boot: int = 10_000, seed: int = 0, include_plotlyjs: bool = True,
                        period_returns=None, block: int = 6, stationary: bool = False) -> str:
    """The whole-history read as a scorecard: the reality-check verdict plus the
    magnitude the verdict is about (total R = E x N, and R per year), then the
    metric strip and the four edge panels.

    Distinct from `gauntlet_report` by design — it leads with stat tiles, not the
    pillar-chip banner — so a full-range run looks like a full-range run. Returns
    the HTML string, or writes it to ``path`` and returns the path."""
    v = reality_check(trades, n_boot=n_boot, seed=seed)
    total_r = float(np.sum(np.asarray(trades.r, dtype=float)))
    years = _years_span(trades)
    color = _VERDICT_COLOR.get(v.label, "#888")

    tiles = [
        _tile(v.label, "verdict &middot; reality check", bg=color),
        _tile(f"{total_r:+.1f} R", "total R (&Sigma;, E&times;N)"),
    ]
    if years:
        tiles.append(_tile(f"{total_r / years:+.1f} R", "R / year"))
    tiles.append(_tile(f"{v.point:+.3f} R",
                       f"expectancy &middot; CI [{v.ci.low:+.2f}, {v.ci.high:+.2f}]"))

    eyebrow = "<div class='cr-eyebrow'>Full range &middot; whole history &middot; reality_check</div>"
    sub = f"<div class='cr-sub'>{subtitle}</div>" if subtitle else ""
    panels = edge_panels(trades, include_plotlyjs=include_plotlyjs, n_boot=n_boot, seed=seed,
                         period_returns=period_returns, block=block, stationary=stationary)
    inner = (f"{title_lockup(title)}{eyebrow}{sub}"
             f"<div class='cr-tiles'>{''.join(tiles)}</div>"
             f"{metrics_table(trades)}{panels}{_FOOT}")
    doc = _page(title, inner)
    if path is None:
        return doc
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def _split_card(role: str, verd, n: int, *, honest: bool, metric_name: str) -> str:
    color = _VERDICT_COLOR.get(verd.label, "#888")
    cls = "cr-splitcard cr-honest" if honest else "cr-splitcard"
    style = f" style='border-color:{color}'" if honest else ""
    note = ("the honest read: data the fit never saw" if honest
            else "context: where an edge would be chosen")
    return (f"<div class='{cls}'{style}>"
            f"<div class='role'>{role} &middot; n={n}</div>"
            f"<div class='verd' style='color:{color}'>{verd.label}</div>"
            f"<div class='line'>{metric_name} {verd.point:+.3f} R &nbsp; "
            f"CI [{verd.ci.low:+.3f}, {verd.ci.high:+.3f}]</div>"
            f"<div class='line'>p(edge&gt;0) = {1 - verd.p_value:.3f}</div>"
            f"<div class='sub'>{note}</div></div>")


def _holdout_cumr_chart(trades: TradeLog, split_ts: pd.Timestamp, embargo_weeks: int,
                        *, include_plotlyjs: bool = False) -> str:
    """Cumulative R of the whole log with the split marked and the embargo band
    shaded — the picture of what TRAIN saw vs the untouched TEST period. ``''`` when
    the log has no dates to place the split on."""
    cr = cumulative_r(trades)
    if not isinstance(cr.index, pd.DatetimeIndex):
        return ""
    go, _ = _plotly()
    grid = "rgba(128,128,128,0.18)"
    emb_end = split_ts + pd.Timedelta(weeks=embargo_weeks)
    fig = go.Figure(go.Scatter(x=list(cr.index), y=cr.values, mode="lines",
                               line=dict(color="#1a7f37", width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.add_vrect(x0=split_ts, x1=emb_end, fillcolor="rgba(154,103,0,0.14)", line_width=0,
                  annotation_text="embargo", annotation_position="bottom right",
                  annotation_font=dict(color="#9a6700", size=10))
    fig.add_vline(x=split_ts, line_dash="dash", line_color="#888",
                  annotation_text="split", annotation_position="top left",
                  annotation_font=dict(color="#5b6570", size=11))
    fig.update_layout(height=300, showlegend=False, margin=dict(t=42, l=52, r=24, b=36),
                      title="Cumulative R — train (pre-split) then the untouched test period",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#5b6570"))
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, title_text="cumulative R")
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def holdout_report(trades: TradeLog, split, path: Optional[str] = None, *,
                   embargo_weeks: int = 8, metric=expectancy, metric_name: str = "expectancy",
                   title: str = "Holdout scorecard", subtitle: Optional[str] = None,
                   n_boot: int = 10_000, seed: int = 0, include_plotlyjs: bool = True) -> str:
    """The early-train / late-confirm split as a scorecard: TRAIN and the untouched
    TEST as two cards (TEST emphasized — it is the verdict), over a cumulative-R
    chart split at the boundary with the embargo band shaded.

    Computes the split with `crucible.validation.holdout` (same leakage controls).
    Returns the HTML string, or writes it to ``path`` and returns the path."""
    from crucible.validation.holdout import holdout as _holdout

    res = _holdout(trades, split, embargo_weeks=embargo_weeks, metric=metric,
                   metric_name=metric_name, n_boot=n_boot, seed=seed)
    split_ts = pd.Timestamp(split)
    vcolor = _VERDICT_COLOR.get(res.label, "#888")

    eyebrow = (f"<div class='cr-eyebrow'>Holdout &middot; early-train / late-confirm "
               f"&middot; split {split_ts.date()} &middot; embargo {embargo_weeks}w</div>")
    sub = f"<div class='cr-sub'>{subtitle}</div>" if subtitle else ""
    cards = (f"<div class='cr-split'>"
             f"{_split_card('Train', res.train, res.train_n, honest=False, metric_name=metric_name)}"
             f"{_split_card('Test', res.test, res.test_n, honest=True, metric_name=metric_name)}"
             f"</div>")
    verdict_line = (f"<div class='cr-sub'>Verdict = the untouched <b>TEST</b> period: "
                    f"<b style='color:{vcolor}'>{res.label}</b></div>")
    chart = _holdout_cumr_chart(trades, split_ts, embargo_weeks, include_plotlyjs=include_plotlyjs)
    inner = f"{title_lockup(title)}{eyebrow}{sub}{cards}{verdict_line}{chart}{_FOOT}"
    doc = _page(title, inner)
    if path is None:
        return doc
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
