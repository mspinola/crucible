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
  .cr-verdict-row {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin: 8px 0 18px; }}
  .cr-verdict-row .cr-verdict, .cr-verdict-row .cr-pillars {{ margin: 0; }}
  .cr-pillars .ok {{ color: var(--cr-pass); font-weight: 600; }}
  .cr-pillars .no {{ color: var(--cr-fail); font-weight: 600; }}
  .cr-pillars .warn {{ color: var(--cr-warn); font-weight: 600; }}
  .cr-pillars .na {{ color: var(--cr-faint); }}
  .cr-cols {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }}
  .cr-summary {{ margin: 6px 0 16px; max-width: 90ch; color: var(--cr-fg); font-size: 14.5px; }}
  .cr-hostnote {{ align-self: flex-start; padding-top: 16px; }}   /* rides under the prose */
  .cr-summary .lead {{ font-weight: 600; }}
  .cr-summary .no {{ color: var(--cr-fail); font-weight: 600; }}
  .cr-summary .ok {{ color: var(--cr-pass); font-weight: 600; }}
  .cr-summary .warn {{ color: var(--cr-warn); font-weight: 600; }}
  .cr-metrics {{ display: flex; flex-wrap: wrap; gap: 12px 26px; margin: 14px 0 18px;
                padding: 12px 0; border-top: 1px solid var(--cr-border);
                border-bottom: 1px solid var(--cr-border); }}
  .cr-metric {{ display: flex; flex-direction: column; gap: 1px; }}
  .cr-metric .v {{ font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums;
                  color: var(--cr-fg); line-height: 1.15; }}
  .cr-metric .k {{ font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
                  color: var(--cr-muted); }}
  /* Two-column verdict top: interpretation prose (left) + a stats card (right). */
  .cr-top {{ display: flex; gap: 30px; align-items: stretch; flex-wrap: wrap; margin: 6px 0 2px; }}
  .cr-top-left {{ flex: 1 1 52%; min-width: 300px; display: flex; flex-direction: column;
                 justify-content: center; }}   /* center prose+note as a group vs the taller card */
  .cr-top-right {{ flex: 1 1 30%; min-width: 236px; }}
  .cr-top-left .cr-summary {{ margin: 0; max-width: none; }}
  .cr-statcard {{ border: 1px solid var(--cr-border); border-radius: 10px;
                 padding: 14px 18px; background: var(--cr-card); }}
  .cr-statcard .cr-metrics {{ border: 0; margin: 0; padding: 0; display: grid;
                 grid-template-columns: 1fr 1fr; gap: 14px 20px; }}
  .cr-div {{ border: 0; border-top: 1px solid var(--cr-border); margin: 20px 0 16px; }}
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
  table.cr-checks td.bul {{ width: 132px; padding-right: 14px; }}
  .cr-cbul {{ display: inline-block; vertical-align: middle; }}
  .cr-foot {{ color: var(--cr-faint); font-size: 12px; margin-top: 20px; }}
  /* Scorecard chrome (fullrange_scorecard / holdout_report) — deliberately unlike
     the gauntlet banner: an eyebrow + big stat tiles, not a pill + pillar chips. */
  .cr-eyebrow {{ font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
               color: var(--cr-muted); font-weight: 600; margin: 2px 0 12px; }}
  .cr-tiles {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 4px 0 18px; }}
  .cr-tile {{ flex: 1 1 132px; min-width: 122px; border: 1px solid var(--cr-border);
            border-radius: 10px; padding: 13px 16px; background: var(--cr-card); }}
  .cr-tile .v {{ font-size: 23px; font-weight: 700; font-variant-numeric: tabular-nums;
               line-height: 1.12; }}
  .cr-tile .k {{ font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
               color: var(--cr-muted); margin-top: 4px; display: block; }}
  .cr-tile.cr-verdict-tile {{ color: #fff; border: 0; }}
  .cr-tile.cr-verdict-tile .k {{ color: rgba(255,255,255,.85); }}
  .cr-split {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 6px 0 16px; }}
  .cr-splitcard {{ flex: 1 1 250px; border: 1px solid var(--cr-border); border-radius: 10px;
                 padding: 14px 18px; background: var(--cr-card); }}
  .cr-splitcard.cr-honest {{ border-width: 2px; }}
  .cr-splitcard .role {{ font-size: 11px; letter-spacing: .07em; text-transform: uppercase;
                       color: var(--cr-muted); }}
  .cr-splitcard .verd {{ font-size: 19px; font-weight: 700; margin: 3px 0 7px; }}
  .cr-splitcard .line {{ font-size: 13.5px; color: var(--cr-fg);
                       font-variant-numeric: tabular-nums; }}
  .cr-splitcard .sub {{ font-size: 12px; color: var(--cr-muted); margin-top: 7px; }}
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


def monthly_r(trades: TradeLog, *, freq: str = "ME") -> pd.Series:
    """Summed R per calendar period (monthly by default) on a GAP-FREE grid, ordered
    by exit date (entry date as a fallback) — the ordered period-return series the
    block bootstrap consumes (`edge_panels(..., period_returns=monthly_r(tl))`).

    Empty periods contribute 0 R (no trades closed → no period return), so serial
    dependence is read on a true calendar timeline rather than a trade-compressed
    one — the point of the block test. Returns an empty Series when the log carries
    no usable dates. ``freq`` is any pandas offset alias ('ME' month-end, 'W' weekly,
    'QE' quarterly); larger periods mean fewer, coarser observations."""
    f = trades.frame
    order = "exit_date" if "exit_date" in f.columns else (
        "entry_date" if "entry_date" in f.columns else None)
    if order is None or len(f) == 0:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(f[order])
    s = pd.Series(f["r"].to_numpy(dtype=float), index=idx).sort_index()
    return s.resample(freq).sum()


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


# REAL/STRONG/DURABLE answer "is the edge real?" — the core question. GENERAL asks
# the separate "does it generalize beyond the markets it was built on?" A book that
# clears the core but not GENERAL has a real, deployable edge whose *scope* is bounded
# — a caveat, not a failed edge — so it reads as scope-limited (amber), not FAIL (red).
_CORE_PILLARS = ("REAL", "STRONG", "DURABLE")


def _verdict_state(gauntlet) -> str:
    """'pass' — every gate that ran passed. 'scope' — the core edge (REAL/STRONG/
    DURABLE) passed and the ONLY failure is GENERAL: real edge, generalization
    unproven. 'fail' — a core gate failed, so the edge itself is in question."""
    if gauntlet.passed:
        return "pass"
    by = {g.name: g for g in gauntlet.gates}
    core = [by[p] for p in _CORE_PILLARS if p in by]
    core_pass = bool(core) and all(g.passed for g in core)
    failed = {g.name for g in gauntlet.gates if not g.passed}
    return "scope" if (core_pass and failed <= {"GENERAL"}) else "fail"


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
    state = _verdict_state(gauntlet)

    if state == "pass":
        lead = (f"<span class='lead ok'>Validated.</span> "
                f"This book {_join(_PILLAR_PASS.get(nm, nm) for nm in passed)} — "
                f"a real, deployable edge on this evidence.")
        return f"<p class='cr-summary'>{lead}</p>"

    if state == "scope":
        # Core edge holds; only GENERAL fell short. Lead with the edge and frame the
        # generalization miss as scope, not a red flag — don't discard a real edge over it.
        lead = "<span class='lead ok'>Validated on its universe.</span> "
        body = (f"This book {_join(_PILLAR_PASS.get(nm, nm) for nm in passed)} — a real, "
                f"deployable edge on the markets it was built on. Its cross-market "
                f"generalization is <span class='warn'>unproven</span> (it "
                f"{_PILLAR_FAIL.get('GENERAL', 'GENERAL')}): read that as scope, not a "
                f"verdict on the edge — trade the set it's proven on rather than "
                f"extrapolating to new markets on faith.")
        return f"<p class='cr-summary'>{lead}{body}</p>"

    # state == "fail": a core gate broke — the edge itself is in question.
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


def _check_bullet_svg(c) -> str:
    """A tiny inline-SVG bullet for one check: a value bar measured against its
    threshold tick, coloured to match the row's result (green pass / red hard-fail /
    amber soft-fail). Inline SVG (not plotly) so a table of them stays cheap, and it
    inherits the report's theme tokens. ``''`` when the check has no numeric
    value/threshold to place."""
    def _num(x):
        try:
            float(x)
            return not isinstance(x, bool)
        except (TypeError, ValueError):
            return False

    t = getattr(c, "threshold", None)
    if not (_num(c.value) and _num(t)):
        return ""
    v, t = float(c.value), float(t)
    hard, passed = bool(getattr(c, "hard", True)), bool(c.passed)
    color = "var(--cr-pass)" if passed else ("var(--cr-fail)" if hard else "var(--cr-warn)")
    W = 120.0
    lo, hi = min(0.0, v, t), max(0.0, v, t)          # bar is measured from 0
    pad = (hi - lo) * 0.10 or 0.1
    span = (hi + pad) - (lo - pad)

    def _x(val):
        return (val - (lo - pad)) / span * W

    x0, xv, tx = _x(0.0), _x(v), _x(t)
    bx, bw = min(x0, xv), max(abs(xv - x0), 1.2)     # ≥1.2px so a hairline still shows
    tip = f"{c.name}: {v:.4g} vs bar {t:g} — {'clears' if passed else 'misses'}"
    return (
        f"<svg class='cr-cbul' width='120' height='14' viewBox='0 0 120 14' "
        f"role='img' aria-label='{tip}'><title>{tip}</title>"
        f"<rect x='0' y='4.5' width='120' height='5' rx='2.5' fill='var(--cr-rule)'/>"
        f"<rect x='{bx:.1f}' y='4.5' width='{bw:.1f}' height='5' rx='2' fill='{color}'/>"
        f"<line x1='{tx:.1f}' y1='1' x2='{tx:.1f}' y2='13' "
        f"stroke='var(--cr-faint)' stroke-width='1.5'/></svg>")


def gate_block(gate, *, title: Optional[str] = None, expanded: Optional[bool] = None,
               extra_html: str = "") -> str:
    """Render one audited :class:`Gate` as a collapsible card: a summary row (name,
    PASS/FAIL badge, one-line gloss) that expands to a row per check (name, value,
    threshold, hard/soft, result). Hard checks drive the gate's verdict; soft
    checks inform and render muted (amber when failing, never red).

    It opens when the gate FAILED and collapses when it passed — so what needs
    attention shows its checks while clean pillars stay tucked away (state-based
    disclosure). Pass ``expanded`` to force it either way; ``title`` overrides the
    default pillar heading. ``extra_html`` is rendered inside the card just above
    the checks table (e.g. a pillar-specific plot like the STRONG CI whiskers)."""
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
            f"<td class='bul'>{_check_bullet_svg(c)}</td>"
            f"<td class='res {res_cls}'>{res_txt}</td></tr>"
        )
    head = ("<tr><th>Check</th><th>Value</th><th>Threshold</th><th></th>"
            "<th style='text-align:right'>Result</th></tr>")
    table = f"<table class='cr-checks'>{head}{''.join(body_rows)}</table>"
    blurb_html = f"<span class='blurb'>{blurb}</span>" if blurb else ""
    return (f"<details class='cr-gate'{' open' if is_open else ''}>"
            f"<summary><span class='gate-h'>{heading}</span> {badge}{blurb_html}</summary>"
            f"{extra_html}{table}</details>")


def _logo_svg(*, size: int = 30, vessel: str = "currentColor", molten: str = "#e0812b",
              up: str = "var(--cr-pass)", down: str = "var(--cr-fail)") -> str:
    """The crucible mark: a tilted foundry ladle pouring molten that casts a rising
    candlestick chart. The vessel adapts to the page (``currentColor``); the molten
    pour is a fixed glow; the candles use the gauntlet's pass/fail colors."""
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 48 46' fill='none' "
        f"role='img' aria-label='crucible' xmlns='http://www.w3.org/2000/svg'>"
        f"<path d='M18.5 12.8 Q22.8 21.5 26.6 29.6 L29.4 28.4 Q25.6 20.4 21.7 11.6 Z' fill='{molten}'/>"
        f"<ellipse cx='32.5' cy='41.2' rx='9.5' ry='1.9' fill='{molten}'/>"
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


def title_lockup(title: str, *, size: int = 34) -> str:
    """The crucible mark + page title as one top-left lockup (the ``.cr-title`` row
    from ``report_css``). The single source of the branded header so every tearsheet
    — crucible's own and any host's — carries the logo, not just the gauntlet page.
    Returns ``''`` for a falsy title so callers can drop the header cleanly."""
    return f"<div class='cr-title'>{_logo_svg(size=size)}<h1>{title}</h1></div>" if title else ""


def verdict_banner(gauntlet, *, title: Optional[str] = None,
                   subtitle: Optional[str] = None, pillar_notes: Optional[dict] = None,
                   note_html: str = "") -> str:
    """Overall gauntlet verdict: a PASS/FAIL banner plus a one-line pillar summary
    (REAL / STRONG / DURABLE / GENERAL, each ✓ passed, ✗ failed, or — not run).

    ``pillar_notes`` maps a pillar name to a short note (HTML allowed) shown in
    place of the em dash for a pillar that did NOT run — e.g.
    ``{'GENERAL': '→ strategy report'}`` on a single-market page where GENERAL is
    assessed at the strategy level."""
    pillar_notes = pillar_notes or {}
    state = _verdict_state(gauntlet)
    # scope-limited (core edge holds, only GENERAL fell short) reads amber, not red —
    # a real edge with bounded scope, not a failed one.
    label, color = {
        "pass": ("GAUNTLET PASS", "#1a7f37"),
        "scope": ("EDGE VALIDATED", "#9a6700"),
        "fail": ("GAUNTLET FAIL", "#b42318"),
    }[state]
    by_name = {g.name: g for g in gauntlet.gates}
    chips = []
    for p in _PILLARS:
        if p in by_name:
            g = by_name[p]
            if g.passed:
                cls, mark = "ok", "✓"
            elif p in _CORE_PILLARS:
                cls, mark = "no", "✗"          # core miss — the edge is in question (red)
            else:
                cls, mark = "warn", "⚠"        # GENERAL miss — a scope caveat (amber)
            chips.append(f"<span class='{cls}'>{p} {mark}</span>")
        else:
            chips.append(f"<span class='na'>{p} {pillar_notes.get(p, '—')}</span>")
    summary = " &nbsp;·&nbsp; ".join(chips)
    head = title_lockup(title)
    sub = f"<div class='cr-sub'>{subtitle}</div>" if subtitle else ""
    pill = (f"<div class='cr-verdict' style='background:{color}'>{label}"
            f"{' <small>scope-limited</small>' if state == 'scope' else ''}</div>")
    # an optional host note (e.g. the net-of-costs badge) rides the right end of the row
    note = f"<div class='cr-hostnote'>{note_html}</div>" if note_html else ""
    return (f"{head}{sub}"
            f"<div class='cr-verdict-row'>"
            f"{pill}"
            f"<div class='cr-pillars'>{summary}</div>"
            f"{note}"
            f"</div>")


def _block_bootstrap_panel(returns, *, block: int = 6, stationary: bool = False,
                           alpha: float = 0.05, n_boot: int = 10_000,
                           seed: int = 0) -> str:
    """The pooled-book honesty panel: the mean of an ordered PERIOD-return series
    with its i.i.d. bootstrap CI (``block=1``) drawn against its block-bootstrap CI
    (``block=k``). Two whiskers on one period-mean axis — same series, same estimator,
    only the block length differs — so the block whisker is *wider* exactly when the
    series is positively autocorrelated (correlated trades exiting on one macro
    shock). That makes the too-tight i.i.d. band the per-trade "Bootstrap expectancy"
    panel shows visibly the optimistic one, and colors the block whisker by the honest
    verdict (CI lower bound > 0 → holds, straddles zero → fragile, point ≤ 0 → fails).

    Distinct from the per-trade panel on purpose: that one resamples individual trades
    and lives in per-trade expectancy (R/trade); this one resamples calendar blocks and
    lives in mean period return (R/period) — different units, its own axis. Returns an
    embeddable Plotly div (plotly.js never inlined here — the page already ships it), or
    ``''`` when the series has fewer than two periods (no block bootstrap possible)."""
    import plotly.graph_objects as go
    from crucible.edge.stats import block_bootstrap_ci, block_bootstrap_pvalue

    r = np.asarray(getattr(returns, "values", returns), dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return ""
    iid = block_bootstrap_ci(r, block=1, n_boot=n_boot, alpha=alpha, seed=seed)
    blk = block_bootstrap_ci(r, block=block, n_boot=n_boot, stationary=stationary,
                             alpha=alpha, seed=seed)
    p = block_bootstrap_pvalue(r, block=block, n_boot=n_boot, stationary=stationary, seed=seed)

    PASS, FAIL, WARN, MUTE = "#1a7f37", "#b42318", "#9a6700", "#8b949e"
    bcolor = FAIL if blk.point <= 0 else (PASS if blk.low > 0 else WARN)
    verdict = "holds" if blk.low > 0 else ("straddles zero" if blk.point > 0 else "fails")
    lvl = int(round((1 - alpha) * 100))

    go_ = go.Figure()
    # block whisker on top (y=1), the i.i.d. reference muted below (y=0).
    for lbl, ci, color, y in ((f"block={block}", blk, bcolor, 1.0),
                              ("i.i.d. (block=1)", iid, MUTE, 0.0)):
        go_.add_trace(go.Scatter(
            x=[ci.point], y=[y], mode="markers",
            marker=dict(color=color, size=11, line=dict(width=1, color="rgba(0,0,0,0.35)")),
            error_x=dict(type="data", symmetric=False, array=[ci.high - ci.point],
                         arrayminus=[ci.point - ci.low], color=color, thickness=2, width=7),
            hovertemplate=(f"{lbl}<br>mean {ci.point:.3f} R/period<br>"
                           f"{lvl}%% CI [{ci.low:.3f}, {ci.high:.3f}]<extra></extra>")))
    go_.add_vline(x=0, line_dash="dot", line_color="rgba(128,128,128,0.6)")
    go_.update_yaxes(tickvals=[0, 1], ticktext=["i.i.d.<br>block=1", f"block<br>={block}"],
                     showgrid=False, zeroline=False, range=[-0.6, 1.6])
    go_.update_xaxes(title_text="mean period return (R)",
                     gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    go_.update_layout(
        height=210, showlegend=False, bargap=0.05,
        title=dict(text=(f"Block bootstrap — honest CI for a clustered book "
                         f"(p={p:.3f}, edge {verdict})"),
                   font=dict(color="#8b949e", size=13), x=0.0, xanchor="left"),
        margin=dict(l=96, r=30, t=42, b=44),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"))
    return go_.to_html(full_html=False, include_plotlyjs=False)


def edge_panels(trades: TradeLog, *, include_plotlyjs: bool = False,
                n_boot: int = 10_000, seed: int = 0,
                period_returns=None, block: int = 6, stationary: bool = False,
                alpha: float = 0.05) -> str:
    """The four capital-free edge panels as an embeddable HTML fragment:
    R-multiple distribution, cumulative R, MFE-vs-MAE excursion (when present),
    and the bootstrap expectancy distribution with its CI and point estimate.

    ``include_plotlyjs=False`` (the default) omits the plotly.js payload so the
    host page can load it once; pass ``True`` for a standalone fragment.

    The "Bootstrap expectancy" panel is the **i.i.d.** trade bootstrap — honest for a
    single instrument, but too tight for a **pooled multi-asset book** whose trades
    cluster in calendar time. Pass ``period_returns`` (an ordered period-return series
    — e.g. ``monthly_r(trades)``) to append a **block-bootstrap panel** that draws the
    correlation-preserving CI against the i.i.d. one on the period-mean axis; ``block``
    (periods), ``stationary`` (Politis–Romano random block lengths) and ``alpha`` tune
    it. Omitted (``None``) leaves the fragment exactly as before."""
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
    html = fig.to_html(full_html=False,
                       include_plotlyjs=(True if include_plotlyjs else "cdn"))
    # Optional block-bootstrap panel: honest CI for a time-clustered pooled book,
    # rendered below the 2×2 grid only when the caller supplies a period series. Its
    # plotly.js is never re-inlined — the main fragment (or the host page) carries it.
    if period_returns is not None:
        html += _block_bootstrap_panel(period_returns, block=block, stationary=stationary,
                                       alpha=alpha, n_boot=n_boot, seed=seed)
    return html


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


def _strong_whisker(gate, trades: TradeLog, *, n_boot: int = 10_000,
                    alpha: float = 0.05, seed: int = 0) -> str:
    """The STRONG pillar rendered as CI whiskers — expectancy, profit factor and SQN,
    each a point estimate + bootstrap CI against its floor (the dashed line). STRONG
    passes a metric when its CI *lower* bound clears the floor, i.e. the whole whisker
    sits right of the line — so a book whose point estimate clears but whose CI crosses
    back (the "PF 1.37 on 60 trades" case) reads as a fail at a glance.

    Floors and pass/fail are read straight from ``gate``'s checks so the picture always
    matches the check table; the point/CI geometry is recomputed with the same bootstrap
    the gate used (deterministic seed → identical numbers). Three stacked panels because
    the metrics live on different scales. Returns an embeddable Plotly div
    (``include_plotlyjs`` is never inlined here — the page already ships plotly.js), or
    ``''`` when the gate or trade log can't supply the three CI checks."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from crucible.validation.gauntlet import bootstrap_metric_cis

    r = getattr(trades, "r", None)
    if gate is None or r is None or len(r) == 0:
        return ""
    by_check = {c.name: c for c in getattr(gate, "checks", [])}
    # (panel label, metric key in the CI dict, gate check name, hard?)
    specs = [("Expectancy (R)", "expectancy", "expectancy_ci_lower", True),
             ("Profit factor", "profit_factor", "profit_factor_ci_lower", True),
             ("SQN-100", "sqn", "sqn_ci_lower", False)]
    cis = bootstrap_metric_cis(trades, n_boot=n_boot, alpha=alpha, seed=seed)
    specs = [s for s in specs if s[2] in by_check and s[1] in cis]
    if not specs:
        return ""

    PASS, HARDFAIL, SOFT = "#1a7f37", "#b42318", "#9a6700"
    fig = make_subplots(rows=len(specs), cols=1, vertical_spacing=0.22,
                        subplot_titles=[f"{lbl}{'' if hard else '  ·  soft'}"
                                        for lbl, _, _, hard in specs])
    for i, (lbl, key, cname, hard) in enumerate(specs, start=1):
        ci = cis[key]
        chk = by_check[cname]
        floor = chk.threshold
        clears = bool(chk.passed)
        color = PASS if clears else (HARDFAIL if hard else SOFT)
        fig.add_trace(go.Scatter(
            x=[ci.point], y=[0], mode="markers+text",
            marker=dict(color=color, size=11, line=dict(width=1, color="rgba(0,0,0,0.35)")),
            error_x=dict(type="data", symmetric=False, array=[ci.high - ci.point],
                         arrayminus=[ci.point - ci.low], color=color, thickness=2, width=6),
            text=[f"  {ci.point:.2f}"], textposition="middle right",
            textfont=dict(color="#8b949e", size=11),
            hovertemplate=(f"{lbl}<br>point {ci.point:.3f}<br>"
                           f"90%% CI [{ci.low:.3f}, {ci.high:.3f}]<br>"
                           f"floor {floor:g} → {'clears' if clears else 'below'}<extra></extra>")),
            row=i, col=1)
        if floor is not None:
            fig.add_vline(x=floor, line_dash="dash", line_color="rgba(148,163,184,0.75)",
                          annotation_text=f"floor {floor:g}", annotation_position="top left",
                          annotation_font=dict(color="#8b949e", size=10), row=i, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                         range=[-1, 1], row=i, col=1)
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, row=i, col=1)
    fig.update_layout(height=90 * len(specs) + 60, margin=dict(l=20, r=30, t=34, b=24),
                      showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e"))
    fig.update_annotations(font=dict(color="#c9d1d9", size=13))  # subplot titles
    return fig.to_html(full_html=False, include_plotlyjs=False)


# Friendly labels for the headline check of each pillar (falls back to the raw name).
_BULLET_LABEL = {
    "permutation_pvalue": "significance p",
    "reality_check_pvalue": "data-mined p",
    "expectancy_ci_lower": "expectancy CI-low (R)",
    "profit_factor_ci_lower": "profit-factor CI-low",
    "sqn_ci_lower": "SQN CI-low",
    "wfe_sqn_aggregate": "walk-forward SQN",
    "fold_dispersion": "fold dispersion",
    "cross_market_reality_check": "cross-market p",
}


def pillar_bullets(gauntlet, *, include_plotlyjs: bool = False) -> str:
    """The four pillars as bullet plots — each pillar's headline (first hard) check as a
    bar measured against its threshold (the dashed line), so *how far* it clears or
    misses reads at a glance: the margin the ✓/✗ chips hide (REAL crushing the bar vs
    GENERAL sitting right on it). Green clears; amber = a GENERAL (scope) miss; red = a
    core REAL/STRONG/DURABLE miss. Each pillar keeps its own scale — the metrics live in
    different units, so bar length is compared to that pillar's own line, not across
    pillars. ``''`` when no gate exposes a numeric headline check.

    Direction is inferred, not mapped: a check is higher-is-better exactly when
    ``(value >= threshold) == passed`` — so p-values (lower better) and SQN/expectancy
    (higher better) both read correctly without a per-metric table."""
    go, make_subplots = _plotly()

    def _num(x):
        # scalar real only — some checks carry a tuple/None value (e.g. a pair) that
        # has no single bar-vs-threshold reading; skip those, don't crash on them.
        try:
            float(x)
            return not isinstance(x, bool)
        except (TypeError, ValueError):
            return False

    rows = []
    for g in gauntlet.gates:
        checks = getattr(g, "checks", None) or []
        hard = [c for c in checks if getattr(c, "hard", False)]
        for c in (hard or checks):
            if _num(c.value) and _num(c.threshold):
                rows.append((g.name, c))
                break
    if not rows:
        return ""

    PASS, WARN, FAIL = "#1a7f37", "#9a6700", "#b42318"
    titles = [f"{name}  ·  {_BULLET_LABEL.get(c.name, c.name.replace('_', ' '))}"
              for name, c in rows]
    # 2-column grid (so four pillars read as a compact 2×2, not a tall 4×1); ≤1 pillar
    # stays single. subplot_titles are row-major and must fill every grid cell.
    n = len(rows)
    ncols = 2 if n >= 2 else 1
    nrows = (n + ncols - 1) // ncols
    grid_titles = [titles[i] if i < n else "" for i in range(nrows * ncols)]
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=(0.30 if nrows > 1 else 0.0),
                        horizontal_spacing=0.13, subplot_titles=grid_titles)
    for i, (name, c) in enumerate(rows):
        rr, cc = i // ncols + 1, i % ncols + 1
        v, t, passed = float(c.value), float(c.threshold), bool(c.passed)
        color = PASS if passed else (WARN if name == "GENERAL" else FAIL)
        higher_better = (v >= t) == passed
        margin = (v - t) if higher_better else (t - v)          # >0 = clears the bar
        lo, hi = min(0.0, v, t), max(0.0, v, t)                 # bar is measured from 0
        pad = (hi - lo) * 0.20 or 0.1
        fig.add_trace(go.Bar(
            x=[v], y=[0], orientation="h", width=0.5,
            marker=dict(color=color, line=dict(width=0)),
            text=[f" {v:.4g}"], textposition="outside", cliponaxis=False,
            textfont=dict(color="#8b949e", size=11),
            hovertemplate=(f"{name}<br>value {v:.4g} · bar {t:g}<br>"
                           f"{'clears' if passed else 'misses'} by {abs(margin):.4g}"
                           f"<extra></extra>")),
            row=rr, col=cc)
        fig.add_vline(x=t, line_dash="dash", line_color="rgba(148,163,184,0.85)",
                      annotation_text=f"bar {t:g}", annotation_position="top left",
                      annotation_font=dict(color="#8b949e", size=10), row=rr, col=cc)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                         range=[-0.8, 0.8], row=rr, col=cc)
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False,
                         range=[lo - pad, hi + pad], row=rr, col=cc)
    fig.update_layout(height=120 * nrows + 46, margin=dict(l=20, r=30, t=30, b=18),
                      showlegend=False, bargap=0.4, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e"))
    fig.update_annotations(font=dict(color="#c9d1d9", size=13))   # subplot titles
    # As the FIRST figure on the gauntlet page, the bullet strip carries plotly.js
    # (inline, or a synchronous CDN tag) so it loads before this figure's own init
    # script — later figures (edge panels, whisker) then find window.Plotly already up.
    return fig.to_html(full_html=False,
                       include_plotlyjs=(True if include_plotlyjs else "cdn"))


# Shown under the verdict when the host supplies no cost attestation. crucible sees only
# the `r` column, so it never assumes the returns are net — it asks.
_COSTS_NOT_ATTESTED = (
    "<span style='display:inline-block;padding:4px 11px;border-radius:999px;"
    "font-size:12px;font-weight:600;background:rgba(210,153,34,0.14);color:var(--cr-warn)'>"
    "costs not attested · is <i>r</i> net of commission + slippage?</span>"
)


def gauntlet_report(gauntlet, trades: TradeLog, path: Optional[str] = None, *,
                    title: str = "crucible gauntlet", subtitle: Optional[str] = None,
                    appendix_html: str = "", header_html: str = "",
                    n_boot: int = 10_000, seed: int = 0,
                    include_plotlyjs: bool = True, pillar_notes: Optional[dict] = None,
                    period_returns=None, block: int = 6, stationary: bool = False) -> str:
    """Compose a full gauntlet-organized page: verdict banner → a cost note that
    sits right under the verdict (``header_html`` if the host supplies one, e.g. a
    net/gross badge; otherwise a default ``costs not attested`` nudge, since crucible
    can't tell a net log from a gross one) → edge panels + metrics → one section per pillar that ran
    (REAL / STRONG / DURABLE / GENERAL; STRONG carries a CI-whisker plot of its
    checks) → an optional host-supplied
    ``appendix_html`` (e.g. capital-aware panels the host owns). ``pillar_notes``
    relabels an un-run pillar in the banner (see `verdict_banner`).

    For a pooled multi-asset book, pass ``period_returns`` (an ordered period-return
    series such as ``monthly_r(trades)``) to add the block-bootstrap honesty panel
    below the edge panels — the CI that survives calendar clustering, drawn against
    the optimistic i.i.d. one (``block`` / ``stationary`` tune it; see `edge_panels`).
    If ``path`` is given the HTML is written there and the path is returned; otherwise
    the HTML string is returned."""
    # the host note (e.g. net-of-costs badge) rides the right end of the verdict row
    banner = verdict_banner(gauntlet, title=title, subtitle=subtitle,
                            pillar_notes=pillar_notes)
    summary = verdict_summary(gauntlet)
    # bullets is the FIRST figure → it carries plotly.js so everything below finds it
    bullets = pillar_bullets(gauntlet, include_plotlyjs=include_plotlyjs)
    panels = edge_panels(trades, include_plotlyjs=include_plotlyjs, n_boot=n_boot, seed=seed,
                         period_returns=period_returns, block=block, stationary=stationary)
    metrics = metrics_table(trades)
    by_name = {g.name: g for g in gauntlet.gates}

    def _block(p):
        # STRONG gets its CI checks drawn as whiskers above the table; recomputed with
        # the report's bootstrap settings so the numbers match the gate's own CIs.
        extra = _strong_whisker(by_name[p], trades, n_boot=n_boot, seed=seed) if p == "STRONG" else ""
        return gate_block(by_name[p], extra_html=extra)

    pillars = "".join(_block(p) for p in _PILLARS if p in by_name)
    appendix = appendix_html or ""
    # Summary text (left) and the host note (e.g. the net-of-costs badge) sit on one row
    # so the note fills the space beside the ≤70ch prose instead of stranding it below.
    # Two-column top: interpretation prose (left) + the stats as a card (right); then a
    # rule before the pillar-margin bullets.
    # the host note (net-of-costs badge) sits at the bottom of the left column so the
    # prose + note balance the taller stats card on the right. With no host note, crucible
    # cannot tell a net log from a gross one — both are just an `r` column — so rather than
    # imply the returns are net it stamps "costs not attested". A host that has netted (and
    # says so via header_html) suppresses this; a bare log gets nudged to declare costs.
    note = f"<div class='cr-hostnote'>{header_html or _COSTS_NOT_ATTESTED}</div>"
    top = (f"<div class='cr-top'>"
           f"<div class='cr-top-left'>{summary}{note}</div>"
           f"<div class='cr-top-right'><div class='cr-statcard'>{metrics}</div></div>"
           f"</div>")
    inner = (f"{banner}<hr class='cr-div'>{top}<hr class='cr-div'>"
             f"{bullets}{panels}{pillars}{appendix}{_FOOT}")
    doc = _page(title, inner)
    if path is None:
        return doc
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def tearsheet(trades: TradeLog, path: str = "tearsheet.html", *,
              title: str = "crucible tearsheet", subtitle: Optional[str] = None,
              n_boot: int = 10_000, seed: int = 0,
              include_plotlyjs: bool = True,
              period_returns=None, block: int = 6, stationary: bool = False) -> str:
    """Write a self-contained single-book HTML tearsheet and return its path.

    Built from the shared blocks: the reality-check verdict banner (HELD /
    FRAGILE / FAIL), the metrics table, and the four edge panels. For a
    gauntlet-organized page (REAL/STRONG/DURABLE/GENERAL) use `gauntlet_report`.
    ``include_plotlyjs=True`` inlines plotly.js so the file renders offline. Pass
    ``period_returns`` (e.g. ``monthly_r(trades)``) to append the block-bootstrap
    honesty panel for a time-clustered book (``block`` / ``stationary`` tune it)."""
    v = reality_check(trades, n_boot=n_boot, seed=seed)
    sub = f"<div class='cr-sub'>{subtitle}</div>" if subtitle else ""
    banner = _reality_banner(v)
    metrics = metrics_table(trades)
    panels = edge_panels(trades, include_plotlyjs=include_plotlyjs, n_boot=n_boot, seed=seed,
                         period_returns=period_returns, block=block, stationary=stationary)
    inner = f"{title_lockup(title)}{sub}{banner}{metrics}{panels}{_FOOT}"
    doc = _page(title, inner)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
