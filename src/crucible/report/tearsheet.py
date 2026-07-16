"""Capital-free HTML report blocks: the edge, its shape, and its verdict.

Two ways to use this module:

* `tearsheet(trades, ...)` — the original one-call, self-contained page built
  around the single-book reality-check verdict (HELD / FRAGILE / FAIL).
* The composable **blocks** — `verdict_banner`, `edge_panels`, `metrics_table`,
  `gate_block`, `report_css`, and the full-page `gauntlet_report` — for
  assembling a gauntlet-organized page (REAL / STRONG / DURABLE / GENERAL) that a
  host application can extend with its own panels.

Everything here visualizes the EDGE, not an account: R-multiples, no capital,
position sizing, or equity curve.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from crucible.edge.trade_log import TradeLog
from crucible.edge.metrics import edge_report, expectancy
from crucible.edge.stats import reality_check

# Reality-check verdict (single book).
_VERDICT_COLOR = {"HELD": "#1a7f37", "FRAGILE": "#9a6700", "FAIL": "#b42318"}

# One line per gauntlet pillar: what passing it proves. Keys are the gate names
# produced by crucible.validation.run_gauntlet.
_PILLARS = ("REAL", "STRONG", "DURABLE", "GENERAL")
_PILLAR_BLURB = {
    "REAL": "The edge is statistically distinguishable from noise.",
    "STRONG": "The edge's lower confidence bounds clear the bar.",
    "DURABLE": "The edge persists out-of-sample, fold after fold.",
    "GENERAL": "The edge holds across more than one market.",
}


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


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
# Theme tokens. Light is the default; a viewer in dark mode gets the dark palette
# via prefers-color-scheme, and a host that stamps data-theme on a wrapping
# element overrides either way (the toggle always wins). Semantic pass/fail/warn
# brighten slightly in the dark palette so they stay legible on a dark surface.
_THEME_LIGHT = """
    --cr-bg:#ffffff; --cr-fg:#1a1a1a; --cr-muted:#666666; --cr-faint:#999999;
    --cr-border:#e6e6e6; --cr-rule:#f2f2f2; --cr-card:#ffffff;
    --cr-pass:#1a7f37; --cr-fail:#b42318; --cr-warn:#9a6700;
    --cr-pass-bg:#e7f4ea; --cr-fail-bg:#fdecea;
"""
_THEME_DARK = """
    --cr-bg:#0f1419; --cr-fg:#e6e6e6; --cr-muted:#9aa4ad; --cr-faint:#6e7781;
    --cr-border:#2a323b; --cr-rule:#232b33; --cr-card:#161c22;
    --cr-pass:#3fb950; --cr-fail:#f85149; --cr-warn:#d29922;
    --cr-pass-bg:#12261a; --cr-fail-bg:#2a1517;
"""


def report_css() -> str:
    """The report's stylesheet (no surrounding ``<style>`` tag). Classes are
    ``cr-`` prefixed so an embedding page's own CSS can't collide. Theme-aware:
    light by default, dark under ``prefers-color-scheme: dark``, and a host may
    force either by setting ``data-theme="light"|"dark"`` on a wrapping element.
    Include once."""
    return f"""
  :root {{ {_THEME_LIGHT} }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ {_THEME_DARK} }} }}
  [data-theme="dark"] {{ {_THEME_DARK} }}
  [data-theme="light"] {{ {_THEME_LIGHT} }}
  .cr-wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px; background: var(--cr-bg);
             font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; color: var(--cr-fg); }}
  .cr-wrap h1 {{ margin: 0 0 2px; font-size: 22px; }}
  .cr-title {{ display: flex; align-items: center; gap: 11px; margin: 0 0 2px; }}
  .cr-title h1 {{ margin: 0; }}
  .cr-title svg {{ flex: none; }}
  .cr-sub {{ color: var(--cr-muted); margin-bottom: 16px; }}
  .cr-verdict {{ display: inline-block; padding: 8px 14px; border-radius: 8px;
                color: #fff; font-weight: 600; margin: 8px 0 4px; }}
  .cr-verdict small {{ font-weight: 400; opacity: .9; }}
  .cr-pillars {{ color: var(--cr-muted); margin: 6px 0 18px; font-size: 14px; }}
  .cr-pillars .ok {{ color: var(--cr-pass); font-weight: 600; }}
  .cr-pillars .no {{ color: var(--cr-fail); font-weight: 600; }}
  .cr-pillars .na {{ color: var(--cr-faint); }}
  .cr-cols {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }}
  .cr-summary {{ margin: 6px 0 16px; max-width: 70ch; color: var(--cr-fg); font-size: 14.5px; }}
  .cr-summary .lead {{ font-weight: 600; }}
  .cr-summary .no {{ color: var(--cr-fail); font-weight: 600; }}
  .cr-summary .ok {{ color: var(--cr-pass); font-weight: 600; }}
  .cr-metrics {{ display: flex; flex-wrap: wrap; gap: 12px 26px; margin: 14px 0 18px;
                padding: 12px 0; border-top: 1px solid var(--cr-border);
                border-bottom: 1px solid var(--cr-border); }}
  .cr-metric {{ display: flex; flex-direction: column; gap: 1px; }}
  .cr-metric .v {{ font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums;
                  color: var(--cr-fg); line-height: 1.15; }}
  .cr-metric .k {{ font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
                  color: var(--cr-muted); }}
  .cr-gate {{ border: 1px solid var(--cr-border); border-radius: 10px;
             margin: 12px 0; background: var(--cr-card); }}
  .cr-gate > summary {{ list-style: none; cursor: pointer; padding: 12px 16px;
             display: flex; align-items: baseline; gap: 6px 12px; flex-wrap: wrap; }}
  .cr-gate > summary::-webkit-details-marker {{ display: none; }}
  .cr-gate > summary::before {{ content: '\\25B8'; color: var(--cr-muted); font-size: 12px;
             align-self: center; }}
  .cr-gate[open] > summary::before {{ content: '\\25BE'; }}
  .cr-gate .gate-h {{ font-size: 16px; font-weight: 600; }}
  .cr-gate .blurb {{ color: var(--cr-muted); font-size: 13px; }}
  .cr-gate table.cr-checks {{ margin: 0 16px 14px; width: calc(100% - 32px); }}
  .cr-tag {{ font-size: 11px; font-weight: 700; letter-spacing: .04em; padding: 2px 8px;
            border-radius: 999px; text-transform: uppercase; }}
  .cr-tag.pass {{ background: var(--cr-pass-bg); color: var(--cr-pass); }}
  .cr-tag.fail {{ background: var(--cr-fail-bg); color: var(--cr-fail); }}
  table.cr-checks {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  table.cr-checks th {{ text-align: left; color: var(--cr-muted); font-weight: 500;
                       border-bottom: 1px solid var(--cr-border); padding: 5px 10px 5px 0; }}
  table.cr-checks td {{ padding: 5px 10px 5px 0; font-variant-numeric: tabular-nums;
                       border-bottom: 1px solid var(--cr-rule); }}
  table.cr-checks td.name {{ font-weight: 500; }}
  table.cr-checks td.res {{ text-align: right; font-weight: 700; }}
  table.cr-checks .hard {{ color: var(--cr-fg); }}
  table.cr-checks .soft {{ color: var(--cr-faint); font-style: italic; }}
  table.cr-checks .cpass {{ color: var(--cr-pass); }}
  table.cr-checks .cfail {{ color: var(--cr-fail); }}
  table.cr-checks .cwarn {{ color: var(--cr-warn); }}
  .cr-foot {{ color: var(--cr-faint); font-size: 12px; margin-top: 20px; }}
"""


# --------------------------------------------------------------------------- #
# Small blocks
# --------------------------------------------------------------------------- #
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


def metrics_table(trades: TradeLog) -> str:
    """Compact capital-free metrics as a horizontal stat strip (Trades, win rate,
    expectancy, PF, payoff, SQN-100, and excursion/exit-efficiency when present) —
    laid out in a row rather than a tall column to save vertical space."""
    rep = edge_report(trades).to_dict()
    rows = [
        ("Trades", f"{rep['n']}"),
        ("Win rate", f"{rep['win_rate'] * 100:.1f}%"),
        ("Expectancy", f"{rep['expectancy']:+.3f} R"),
        ("Profit factor", f"{rep['profit_factor']:.2f}"),
        ("Payoff ratio", f"{rep['payoff_ratio']:.2f}"),
        ("SQN-100", f"{rep['sqn']:.2f}"),
    ]
    def _present(x):  # skip absent (None) and degenerate (NaN) metrics
        return x is not None and not (isinstance(x, float) and x != x)
    if _present(rep.get("excursion_ratio")):
        rows.append(("Excursion ratio", f"{rep['excursion_ratio']:.2f}"))
    if _present(rep.get("exit_efficiency")):
        rows.append(("Exit efficiency", f"{rep['exit_efficiency'] * 100:.1f}%"))
    cells = "".join(f"<div class='cr-metric'><span class='v'>{v}</span>"
                    f"<span class='k'>{k}</span></div>" for k, v in rows)
    return f"<div class='cr-metrics'>{cells}</div>"


# Plain-English gloss for each pillar, keyed by gate name — the interpretation
# behind the pass/fail chip.
_PILLAR_PASS = {
    "REAL": "is statistically distinguishable from noise",
    "STRONG": "clears its confidence-interval floors",
    "DURABLE": "persists out-of-sample, fold to fold",
    "GENERAL": "travels across more than one market",
}
_PILLAR_FAIL = {
    "REAL": "isn't distinguishable from random entries — there may be no edge here",
    "STRONG": "clears significance but its confidence floors sit below the bar — the edge is real but thin",
    "DURABLE": "doesn't hold up out-of-sample — it may be period- or regime-specific",
    "GENERAL": "doesn't generalize across markets once you correct for how many were tried",
}


def verdict_summary(gauntlet) -> str:
    """A one-line plain-English reading of the gauntlet: the overall verdict plus
    what passing/failing the pillars that ran means for the book. Rendered under
    the banner so a reader gets the interpretation without decoding the gate
    tables. Speaks only to pillars that actually ran."""
    gates = list(gauntlet.gates)
    if not gates:
        return ""
    passed = [g.name for g in gates if g.passed]
    failed = [g.name for g in gates if not g.passed]

    if gauntlet.passed:
        lead = (f"<span class='lead ok'>Validated.</span> "
                f"This book {_join(_PILLAR_PASS.get(nm, nm) for nm in passed)} — "
                f"a real, deployable edge on this evidence.")
        return f"<p class='cr-summary'>{lead}</p>"

    # The banner + pillar chips already carry the PASS/FAIL count and which pillar
    # broke, so the summary drops that enumeration and keeps only the plain-English
    # reading — what passing/failing actually means for the book.
    lead = "<span class='lead no'>Not validated.</span> "
    if passed:
        body = (f"It {_join(_PILLAR_PASS.get(nm, nm) for nm in passed)}, but "
                f"{_join(_PILLAR_FAIL.get(nm, nm) for nm in failed)}.")
    else:
        body = f"It {_join(_PILLAR_FAIL.get(nm, nm) for nm in failed)}."
    return f"<p class='cr-summary'>{lead}{body}</p>"


def _join(items) -> str:
    """Grammatical join: 'a', 'a and b', 'a, b and c'."""
    xs = list(items)
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _fmt(x) -> str:
    """Format a check value/threshold: numbers to 3 sig-ish figures, None → em dash."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf:  # NaN
        return "—"
    return f"{xf:.3f}"


def gate_block(gate, *, title: Optional[str] = None, expanded: Optional[bool] = None) -> str:
    """Render one audited :class:`Gate` as a collapsible card: a summary row (name,
    PASS/FAIL badge, one-line gloss) that expands to a row per check (name, value,
    threshold, hard/soft, result). Hard checks drive the gate's verdict; soft
    checks inform and render muted (amber when failing, never red).

    It opens when the gate FAILED and collapses when it passed — so what needs
    attention shows its checks while clean pillars stay tucked away (state-based
    disclosure). Pass ``expanded`` to force it either way; ``title`` overrides the
    default pillar heading."""
    name = getattr(gate, "name", "GATE")
    heading = title or name.title()
    blurb = _PILLAR_BLURB.get(name, "")
    passed = bool(gate.passed)
    is_open = (not passed) if expanded is None else bool(expanded)
    badge = (f"<span class='cr-tag {'pass' if passed else 'fail'}'>"
             f"{'pass' if passed else 'fail'}</span>")

    body_rows = []
    for c in gate.checks:
        hard = bool(getattr(c, "hard", True))
        cpass = bool(c.passed)
        # Soft failures are warnings, not failures — they never gate the verdict.
        res_cls = "cpass" if cpass else ("cfail" if hard else "cwarn")
        res_txt = "PASS" if cpass else ("FAIL" if hard else "warn")
        cmp = f"vs {_fmt(c.threshold)}" if getattr(c, "threshold", None) is not None else ""
        detail = getattr(c, "detail", None)
        note = f"<div class='soft' style='font-size:12px'>{detail}</div>" if detail else ""
        body_rows.append(
            f"<tr><td class='name {'hard' if hard else 'soft'}'>{c.name}"
            f"{'' if hard else ' <span class=soft>(soft)</span>'}{note}</td>"
            f"<td>{_fmt(c.value)}</td><td class='{'hard' if hard else 'soft'}'>{cmp}</td>"
            f"<td class='res {res_cls}'>{res_txt}</td></tr>"
        )
    head = ("<tr><th>Check</th><th>Value</th><th>Threshold</th>"
            "<th style='text-align:right'>Result</th></tr>")
    table = f"<table class='cr-checks'>{head}{''.join(body_rows)}</table>"
    blurb_html = f"<span class='blurb'>{blurb}</span>" if blurb else ""
    return (f"<details class='cr-gate'{' open' if is_open else ''}>"
            f"<summary><span class='gate-h'>{heading}</span> {badge}{blurb_html}</summary>"
            f"{table}</details>")


def _logo_svg(*, size: int = 30, vessel: str = "currentColor", molten: str = "#e0812b",
              up: str = "var(--cr-pass)", down: str = "var(--cr-fail)") -> str:
    """The crucible mark: a tilted foundry ladle pouring molten that casts a rising
    candlestick chart. The vessel adapts to the page (``currentColor``); the molten
    pour is a fixed glow; the candles use the gauntlet's pass/fail colors."""
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 48 46' fill='none' "
        f"role='img' aria-label='crucible' xmlns='http://www.w3.org/2000/svg'>"
        f"<path d='M18.5 12.8 Q22.8 21.5 26.6 29.6 L29.4 28.4 Q25.6 20.4 21.7 11.6 Z' fill='{molten}'/>"
        f"<ellipse cx='27.5' cy='40.6' rx='4.6' ry='1.3' fill='{molten}'/>"
        f"<path d='M8 9.5 L5.8 19 Q10.8 26.4 17 23.2 L20.4 13.6' stroke='{vessel}' stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/>"
        f"<path d='M8 9.5 L20.4 13.6' stroke='{vessel}' stroke-width='1.5' stroke-linecap='round'/>"
        f"<path d='M7.2 14 Q11.6 20.4 15.6 17.8' stroke='{vessel}' stroke-width='1.3' stroke-linecap='round'/>"
        f"<line x1='27.5' y1='33' x2='27.5' y2='41' stroke='{down}' stroke-width='1.6' stroke-linecap='round'/>"
        f"<rect x='25.8' y='35' width='3.4' height='4' rx='1' fill='{down}'/>"
        f"<line x1='32.5' y1='29' x2='32.5' y2='40' stroke='{up}' stroke-width='1.6' stroke-linecap='round'/>"
        f"<rect x='30.8' y='31' width='3.4' height='5' rx='1' fill='{up}'/>"
        f"<line x1='37.5' y1='25' x2='37.5' y2='39' stroke='{up}' stroke-width='1.6' stroke-linecap='round'/>"
        f"<rect x='35.8' y='27' width='3.4' height='5' rx='1' fill='{up}'/>"
        f"</svg>"
    )


def _favicon_href() -> str:
    """The mark as an SVG data-URI for `<link rel=icon>` — fixed colors (favicons get
    no page context, so no currentColor / CSS vars)."""
    from urllib.parse import quote
    svg = _logo_svg(size=32, vessel="#7a808a", molten="#e0812b", up="#1a7f37", down="#b42318")
    return "data:image/svg+xml," + quote(svg)


def verdict_banner(gauntlet, *, title: Optional[str] = None,
                   subtitle: Optional[str] = None, pillar_notes: Optional[dict] = None) -> str:
    """Overall gauntlet verdict: a PASS/FAIL banner plus a one-line pillar summary
    (REAL / STRONG / DURABLE / GENERAL, each ✓ passed, ✗ failed, or — not run).

    ``pillar_notes`` maps a pillar name to a short note (HTML allowed) shown in
    place of the em dash for a pillar that did NOT run — e.g.
    ``{'GENERAL': '→ strategy report'}`` on a single-market page where GENERAL is
    assessed at the strategy level."""
    pillar_notes = pillar_notes or {}
    passed = bool(gauntlet.passed)
    color = "#1a7f37" if passed else "#b42318"
    by_name = {g.name: g for g in gauntlet.gates}
    chips = []
    for p in _PILLARS:
        if p in by_name:
            ok = by_name[p].passed
            chips.append(f"<span class='{'ok' if ok else 'no'}'>{p} "
                         f"{'✓' if ok else '✗'}</span>")
        else:
            chips.append(f"<span class='na'>{p} {pillar_notes.get(p, '—')}</span>")
    summary = " &nbsp;·&nbsp; ".join(chips)
    head = (f"<div class='cr-title'>{_logo_svg(size=34)}<h1>{title}</h1></div>") if title else ""
    sub = f"<div class='cr-sub'>{subtitle}</div>" if subtitle else ""
    label = "PASS" if passed else "FAIL"
    return (f"{head}{sub}"
            f"<div class='cr-verdict' style='background:{color}'>GAUNTLET {label}</div>"
            f"<div class='cr-pillars'>{summary}</div>")


def edge_panels(trades: TradeLog, *, include_plotlyjs: bool = False,
                n_boot: int = 10_000, seed: int = 0) -> str:
    """The four capital-free edge panels as an embeddable HTML fragment:
    R-multiple distribution, cumulative R, MFE-vs-MAE excursion (when present),
    and the bootstrap expectancy distribution with its CI and point estimate.

    ``include_plotlyjs=False`` (the default) omits the plotly.js payload so the
    host page can load it once; pass ``True`` for a standalone fragment."""
    go, make_subplots = _plotly()
    r = trades.r
    v = reality_check(trades, n_boot=n_boot, seed=seed)

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
    # R-multiple histogram. A fat right tail (a few +80R winners on a trend/
    # countertrend book) otherwise stretches the axis until the bulk near 0 is an
    # illegible sliver — so clamp values beyond the 99th percentile into an edge
    # bin, hold the view to the bulk, and annotate the overflow with the true max.
    if len(r):
        hi = max(float(np.quantile(r, 0.99)), 5.0)
        lo = min(float(r.min()), -3.0)
        r_disp = np.clip(r, lo, hi)
        n_over = int((r > hi).sum())
    else:
        r_disp, lo, hi, n_over = r, -3.0, 5.0, 0
    fig.add_trace(go.Histogram(x=r_disp, nbinsx=40, marker_color="#4c78a8", name="R"),
                  row=1, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#888", row=1, col=1)
    fig.update_xaxes(range=[lo - 0.5, hi + 0.5], row=1, col=1)
    if n_over:
        fig.add_annotation(row=1, col=1, xref="x domain", yref="y domain",
                           x=0.98, y=0.95, xanchor="right", showarrow=False,
                           font=dict(color="#8b949e", size=11),
                           text=f"▸ {n_over} > {hi:.0f}R (max {float(r.max()):.0f}R)")

    cr = cumulative_r(trades)
    fig.add_trace(go.Scatter(x=list(cr.index), y=cr.values, mode="lines",
                             line_color="#1a7f37", name="cum R"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#888", row=1, col=2)

    mfe, mae = trades.col("mfe"), trades.col("mae")
    if mfe is not None and mae is not None:
        win = r > 0
        fig.add_trace(go.Scatter(x=mae[win], y=mfe[win], mode="markers",
                                 marker=dict(color="#1a7f37", size=6, opacity=0.6),
                                 name="win"), row=2, col=1)
        fig.add_trace(go.Scatter(x=mae[~win], y=mfe[~win], mode="markers",
                                 marker=dict(color="#b42318", size=6, opacity=0.6),
                                 name="loss"), row=2, col=1)

    if len(boot):
        fig.add_trace(go.Histogram(x=boot, nbinsx=40, marker_color="#b0b0b0",
                                   name="boot E"), row=2, col=2)
        for xval, dash, color in ((0, "dot", "#888"),
                                  (v.ci.low, "dash", "#9a6700"),
                                  (v.ci.high, "dash", "#9a6700"),
                                  (v.point, "solid", "#1a7f37")):
            fig.add_vline(x=xval, line_dash=dash, line_color=color, row=2, col=2)

    # Theme-neutral: transparent backgrounds so the chart inherits the page
    # (light or dark), with mid-gray font/gridlines legible on either. Marker
    # colors above already read on both.
    grid = "rgba(128,128,128,0.18)"
    fig.update_layout(height=760, showlegend=False, bargap=0.05,
                      margin=dict(t=48, l=48, r=24, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#8b949e"))
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_annotations(font=dict(color="#8b949e"))  # subplot titles
    return fig.to_html(full_html=False,
                       include_plotlyjs=(True if include_plotlyjs else "cdn"))


def _reality_banner(v) -> str:
    """The single-book reality-check verdict banner (HELD / FRAGILE / FAIL)."""
    color = _VERDICT_COLOR.get(v.label, "#888")
    return (f"<div class='cr-verdict' style='background:{color}'>{v.label} &nbsp;"
            f"<small>{v.metric} {v.point:+.3f} R &nbsp; CI [{v.ci.low:+.3f}, "
            f"{v.ci.high:+.3f}] &nbsp; p(edge&gt;0)={1 - v.p_value:.3f}</small></div>")


# --------------------------------------------------------------------------- #
# Full-page compositions
# --------------------------------------------------------------------------- #
def _page(title: str, inner: str) -> str:
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{title}</title>"
            f"<link rel=\"icon\" href=\"{_favicon_href()}\">"
            f"<style>body{{margin:0;background:var(--cr-bg);}}{report_css()}</style></head>"
            f"<body><div class=\"cr-wrap\">{inner}</div></body></html>")


_FOOT = ("<div class='cr-foot'>crucible — capital-free edge evaluation. Returns in "
         "R-multiples; no capital, position sizing, or equity curve.</div>")


def gauntlet_report(gauntlet, trades: TradeLog, path: Optional[str] = None, *,
                    title: str = "crucible gauntlet", subtitle: Optional[str] = None,
                    appendix_html: str = "", header_html: str = "",
                    n_boot: int = 10_000, seed: int = 0,
                    include_plotlyjs: bool = True, pillar_notes: Optional[dict] = None) -> str:
    """Compose a full gauntlet-organized page: verdict banner → optional
    ``header_html`` (a host note that sits right under the verdict, e.g. a
    net/gross badge) → edge panels + metrics → one section per pillar that ran
    (REAL / STRONG / DURABLE / GENERAL) → an optional host-supplied
    ``appendix_html`` (e.g. capital-aware panels the host owns). ``pillar_notes``
    relabels an un-run pillar in the banner (see `verdict_banner`). If ``path`` is
    given the HTML is written there and the path is returned; otherwise the HTML
    string is returned."""
    banner = verdict_banner(gauntlet, title=title, subtitle=subtitle, pillar_notes=pillar_notes)
    summary = verdict_summary(gauntlet)
    panels = edge_panels(trades, include_plotlyjs=include_plotlyjs, n_boot=n_boot, seed=seed)
    metrics = metrics_table(trades)
    by_name = {g.name: g for g in gauntlet.gates}
    pillars = "".join(gate_block(by_name[p]) for p in _PILLARS if p in by_name)
    appendix = appendix_html or ""
    inner = f"{banner}{metrics}{summary}{header_html or ''}{panels}{pillars}{appendix}{_FOOT}"
    doc = _page(title, inner)
    if path is None:
        return doc
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def tearsheet(trades: TradeLog, path: str = "tearsheet.html", *,
              title: str = "crucible tearsheet", subtitle: Optional[str] = None,
              n_boot: int = 10_000, seed: int = 0,
              include_plotlyjs: bool = True) -> str:
    """Write a self-contained single-book HTML tearsheet and return its path.

    Built from the shared blocks: the reality-check verdict banner (HELD /
    FRAGILE / FAIL), the metrics table, and the four edge panels. For a
    gauntlet-organized page (REAL/STRONG/DURABLE/GENERAL) use `gauntlet_report`.
    ``include_plotlyjs=True`` inlines plotly.js so the file renders offline."""
    v = reality_check(trades, n_boot=n_boot, seed=seed)
    sub = f"<div class='cr-sub'>{subtitle}</div>" if subtitle else ""
    banner = _reality_banner(v)
    metrics = metrics_table(trades)
    panels = edge_panels(trades, include_plotlyjs=include_plotlyjs, n_boot=n_boot, seed=seed)
    inner = f"<h1>{title}</h1>{sub}{banner}{metrics}{panels}{_FOOT}"
    doc = _page(title, inner)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
