"""crucible.ml.decay — does a higher score mean a better outcome?

Bucket a score into equal-count quantiles and read the realized win rate per
bucket. A genuine, well-ordered edge makes win rate climb monotonically from the
worst quantile to the best; a flat or ragged profile is the tell of a score that
ranks nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecayTable:
    """Per-quantile outcome of a score.

    ``table`` has one row per quantile with ``win_rate`` / ``avg_score`` /
    ``count``; ``monotonic`` is True when win rate is non-decreasing across every
    quantile (the signature of a real, well-ordered edge).
    """

    table: pd.DataFrame
    monotonic: bool
    q: int

    @property
    def spread(self) -> float:
        """Top-quantile win rate minus bottom — the crude 'does it separate' number."""
        return float(self.table["win_rate"].iloc[-1] - self.table["win_rate"].iloc[0])


def quantile_decay(preds: pd.DataFrame, *, score: str = "score",
                   label: str = "label", q: int = 5) -> DecayTable:
    """Bucket ``score`` into ``q`` equal-count quantiles and report, per bucket,
    the realized win rate (fraction with ``label > 0``), the average score, and the
    count. A genuine edge makes win rate rise monotonically from Q1 to Q``q``.

    ``win`` is defined as ``label > 0``, so it reads correctly whether losers are
    encoded as ``-1`` or ``0`` — the ambiguity that made the original 'alphalens'
    tearsheet fall back mid-computation. Scores are ranked before bucketing so
    exact-duplicate scores don't collapse a quantile.
    """
    if score not in preds.columns or label not in preds.columns:
        raise ValueError(f"preds needs '{score}' and '{label}' columns")
    df = preds[[score, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < q:
        raise ValueError(f"need at least q={q} valid rows, got {len(df)}")

    df = df.assign(
        _win=(df[label] > 0).astype(float),
        _q=pd.qcut(df[score].rank(method="first"), q, labels=range(1, q + 1)),
    )
    table = (df.groupby("_q", observed=True)
               .agg(win_rate=("_win", "mean"),
                    avg_score=(score, "mean"),
                    count=("_win", "size"))
               .reset_index()
               .rename(columns={"_q": "quantile"}))
    monotonic = bool(np.all(np.diff(table["win_rate"].to_numpy()) >= 0))
    return DecayTable(table=table, monotonic=monotonic, q=q)


def score_by_outcome(preds: pd.DataFrame, *, score: str = "score", label: str = "label",
                     title: str = "Score distribution — winners vs losers",
                     include_plotlyjs: bool | str = False) -> str:
    """Violin distributions of ``score`` split by realized outcome — winners
    (``label > 0``) vs losers (``label <= 0``). The visual companion to the IC and
    quantile-decay numbers: a score that ranks outcomes shows the winners' violin
    sitting above the losers'. Capital-free and model-agnostic like the rest of
    crucible.ml — any ``(score, label)`` predictions frame.

    Returns an embeddable Plotly div (``include_plotlyjs`` passes straight to plotly's
    ``to_html``: ``False`` = script-less for a page that already loads plotly.js,
    ``"cdn"`` / ``True`` for a lone fragment), or ``''`` when no valid rows survive.
    Needs plotly (the ``report`` extra).
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as e:                      # pragma: no cover
        raise ModuleNotFoundError(
            'score_by_outcome needs plotly — install the report extra: '
            'pip install "crucible-quant[report]"') from e
    if score not in preds.columns or label not in preds.columns:
        raise ValueError(f"preds needs '{score}' and '{label}' columns")
    df = preds[[score, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return ""
    fig = go.Figure()
    fig.add_trace(go.Violin(y=df.loc[df[label] > 0, score], name="winners",
                            box_visible=True, meanline_visible=True))
    fig.add_trace(go.Violin(y=df.loc[df[label] <= 0, score], name="losers",
                            box_visible=True, meanline_visible=True))
    fig.update_layout(title=title, yaxis_title="score", height=380,
                      margin=dict(l=40, r=40, t=60, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def decay_tearsheet(preds: pd.DataFrame, *, score: str = "score", label: str = "label",
                    q: int = 5, title: str = "Signal decay by score quantile",
                    out_path: str | None = None) -> str:
    """Render a self-contained HTML tearsheet of a score's decay and return the HTML.

    Two panels: realized win rate by score quantile (from :func:`quantile_decay` —
    a real edge climbs monotonically), and a winners-vs-losers distribution of the
    score. Writes the HTML to ``out_path`` as well when given.

    Needs plotly (the ``report`` extra): ``pip install "crucible-quant[report]"``.
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as e:                      # pragma: no cover
        raise ModuleNotFoundError(
            'decay_tearsheet needs plotly — install the report extra: '
            'pip install "crucible-quant[report]"') from e

    decay = quantile_decay(preds, score=score, label=label, q=q)
    stats = decay.table

    bars = go.Figure()
    bars.add_hline(y=0.5, line_dash="dash", line_color="rgba(128,128,128,0.4)",
                   annotation_text="Coin flip (50%)")
    bars.add_trace(go.Bar(x=[f"Q{int(qq)}" for qq in stats["quantile"]],
                          y=stats["win_rate"],
                          text=[f"{v * 100:.1f}%" for v in stats["win_rate"]],
                          textposition="auto"))
    bars.update_layout(title="Win rate by score quantile (worst → best)",
                       yaxis=dict(tickformat=".0%", title="realized win rate"),
                       xaxis_title="score quantile", height=380,
                       margin=dict(l=40, r=40, t=60, b=40))

    dist_html = score_by_outcome(preds, score=score, label=label, include_plotlyjs=False)

    verdict = "monotonic ✓" if decay.monotonic else "not monotonic"
    pretty = stats.assign(win_rate=(stats["win_rate"] * 100).round(1))
    html = (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>'
        '<body style="font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;">'
        f"<h1>{title}</h1>"
        f"<p>N = {int(stats['count'].sum())} &middot; quantiles = {decay.q} &middot; "
        f"spread (top &minus; bottom) = {decay.spread * 100:.1f}pp &middot; {verdict}</p>"
        + bars.to_html(full_html=False, include_plotlyjs="cdn")
        + dist_html
        + pretty.to_html(index=False)
        + "</body></html>"
    )

    if out_path is not None:
        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html
