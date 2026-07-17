#!/usr/bin/env python3
"""Regenerate the tutorial's figures (docs/img/*.png).

A maintainer helper, NOT part of the docs build — the committed PNGs are what the
site and PDF use. Run it when a worked example changes or a diagram needs an edit.

Rendered from ``crucible.report`` blocks, so they never drift from the numbers:
  gauntlet_hero/scope/gates/cumr/bootstrap.png  the §12 Donchian run (and §11 scope)
  ml_decay.png / ml_verdict.png                 the §13 ML take/skip example

Hand-authored HTML/SVG explainers, kept below as the editable source of truth:
  triple_barrier.png  the triple-barrier labeling method (§1, §7)
  gate_ladder.png     the gauntlet gate ladder (§11)
  meta_label.png      the meta-labeling take/skip pipeline (§7)
  purge_embargo.png   the purge/embargo train-test split (§8)

Plus the crucible mark in two colorways (crucible_logo{,_white}.png).

Requirements (all outside the docs build): the ``[report]`` extra (plotly),
Pillow, and Google Chrome for headless HTML→PNG. Usage::

    pip install "crucible-quant[report]" pillow
    python docs/gen_figures.py            # -> docs/img/*.png (light theme, 2x)
"""
from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMG = Path(__file__).resolve().parent / "img"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# --------------------------------------------------------------------------- #
# Hand-authored explainer diagrams (light theme; the editable source).
# --------------------------------------------------------------------------- #
GATE_LADDER_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
  :root{--ink:#17242b;--mut:#6b7680;--teal:#00695c;--tealbg:#e2f1ef;--tealbd:#8fc9c1;
        --grey:#eef1f2;--greybd:#d3d9dc;--pass:#1a7f37;--fail:#b42318;--bg:#ffffff;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink)}
  .wrap{max-width:760px;margin:0 auto;padding:26px 26px 22px}
  .band-label{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
     margin:16px 2px 8px;font-weight:600}
  .band-label:first-child{margin-top:0}
  .row{display:flex;gap:14px;align-items:baseline;border-radius:10px;padding:12px 16px;margin:8px 0}
  .row .name{font-weight:700;font-size:16px;min-width:118px;letter-spacing:.02em}
  .row .desc{font-size:14px;color:#38454c;line-height:1.4}
  .row .tag{font-size:11px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.08em}
  .pre{background:var(--grey);border:1px solid var(--greybd)}
  .pre .name{color:#41515a}
  .core{background:var(--tealbg);border:1px solid var(--tealbd)}
  .core .name{color:var(--teal)}
  .opt{opacity:.82}
  .hand{background:repeating-linear-gradient(135deg,#fafafa,#fafafa 8px,#f3f3f3 8px,#f3f3f3 16px);
        border:1px dashed var(--greybd)}
  .hand .name{color:#5a666e}
  .rule{display:flex;gap:12px;align-items:center;margin-top:18px;padding:13px 16px;border-radius:10px;
        background:#fdecea;border:1px solid #f3c0b8;color:#7a1c12;font-size:14px;line-height:1.4}
  .rule .loop{font-size:22px;line-height:1}
  .rule b{color:var(--fail)}
  h2{margin:0 0 4px;font-size:15px;color:var(--teal);letter-spacing:.02em}
  .sub{font-size:12.5px;color:var(--mut);margin-bottom:4px}
</style></head><body><div class="wrap">
  <h2>The gauntlet — an ordered set of audited hard gates</h2>
  <div class="sub">Each gate is an AND of its hard checks. No gate is skippable; a strong later gate can't redeem an early fail.</div>

  <div class="band-label">Preambles — you assert these</div>
  <div class="row pre"><div class="name">DECLARE</div><div class="desc">a mechanical rule + a log of every variant you tried</div></div>
  <div class="row pre"><div class="name">CLEAN</div><div class="desc">leakage-controlled construction (holdout / walk-forward)</div></div>

  <div class="band-label">The gauntlet crucible computes</div>
  <div class="row core"><div class="name">REAL</div><div class="desc">distinguishable from noise, corrected for the search</div></div>
  <div class="row core"><div class="name">STRONG</div><div class="desc">economically meaningful at the CI lower bound</div></div>
  <div class="row core"><div class="name">DURABLE</div><div class="desc">holds out-of-sample, fold after fold, over time</div></div>
  <div class="row core opt"><div class="name">GENERAL</div><div class="desc">travels to markets it wasn't built on <span class="tag">optional</span></div></div>

  <div class="band-label">Handoff</div>
  <div class="row hand"><div class="name">SURVIVE</div><div class="desc">capital survivability — <b style="color:#7a1c12">out of scope</b>; hand the surviving log to a position-sizing tool</div></div>

  <div class="rule"><span class="loop">&#10227;</span><div>The non-negotiable rule: a <b>FAIL</b> sends you back to <b>DECLARE</b> — <b>never</b> to tweaking the failing number. That is the anti-data-mining discipline made procedural.</div></div>
</div></body></html>"""

META_LABEL_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
  :root{--ink:#17242b;--mut:#6b7680;--teal:#00695c;--tealbg:#e2f1ef;--tealbd:#8fc9c1;
        --grey:#eef1f2;--greybd:#d3d9dc;--bg:#ffffff;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink)}
  .wrap{max-width:640px;margin:0 auto;padding:26px 26px 22px}
  h2{margin:0 0 4px;font-size:15px;color:var(--teal);letter-spacing:.02em}
  .sub{font-size:12.5px;color:var(--mut);margin-bottom:14px;line-height:1.45}
  .step{display:flex;gap:14px;align-items:baseline;border-radius:10px;padding:12px 16px}
  .step .name{font-weight:700;font-size:15px;min-width:150px;letter-spacing:.02em}
  .step .desc{font-size:13.5px;color:#38454c;line-height:1.4}
  .step .tag{font-size:11px;color:var(--mut);font-weight:600}
  .grey{background:var(--grey);border:1px solid var(--greybd)} .grey .name{color:#41515a}
  .teal{background:var(--tealbg);border:1px solid var(--tealbd)} .teal .name{color:var(--teal)}
  .arrow{color:#9aa4ab;font-size:18px;text-align:center;margin:3px 0 3px 60px;line-height:1}
</style></head><body><div class="wrap">
  <h2>Meta-labeling — a take/skip filter on a primary signal</h2>
  <div class="sub">The primary signal decides <em>which</em> trades exist; the model decides <em>which of them to take</em>. crucible.ml judges the model's score before it reaches a backtester.</div>
  <div class="step grey"><div class="name">Primary signal</div><div class="desc">fires often &mdash; many candidate trades, each with a direction</div></div>
  <div class="arrow">&darr;</div>
  <div class="step grey"><div class="name">Triple-barrier label <span class="tag">&sect;1</span></div><div class="desc">each candidate runs to its barrier &rarr; a realized win / loss</div></div>
  <div class="arrow">&darr;</div>
  <div class="step teal"><div class="name">Model score</div><div class="desc">a take / skip confidence for every candidate</div></div>
  <div class="arrow">&darr;</div>
  <div class="step teal"><div class="name">crucible.ml <span class="tag">&sect;7</span></div><div class="desc">IC &middot; alpha-gate &middot; quantile decay &mdash; <b>is the score real?</b></div></div>
  <div class="arrow">&darr;</div>
  <div class="step teal"><div class="name">Filtered book <span class="tag">&sect;13</span></div><div class="desc">keep only the high scores &rarr; put that book through the gauntlet</div></div>
</div></body></html>"""

TRIPLE_BARRIER_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
  body{margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  .wrap{max-width:820px;margin:0 auto;padding:22px}
  h2{margin:0 0 2px;font-size:15px;color:#00695c}
  .sub{font-size:12.5px;color:#6b7680;margin-bottom:6px}
  text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
</style></head><body><div class="wrap">
  <h2>The triple-barrier method — how an entry becomes a labeled R-multiple</h2>
  <div class="sub">Barriers are sized off the signal bar (no look-ahead); the exit is whichever barrier the path touches first.</div>
  <svg viewBox="0 0 820 400" width="100%" xmlns="http://www.w3.org/2000/svg">
    <line x1="120" y1="40" x2="120" y2="330" stroke="#c7ced2" stroke-width="1"/>
    <line x1="120" y1="330" x2="770" y2="330" stroke="#c7ced2" stroke-width="1"/>
    <line x1="120" y1="80" x2="700" y2="80" stroke="#1a7f37" stroke-width="2"/>
    <text x="128" y="72" fill="#1a7f37" font-size="13" font-weight="600">profit barrier  +tp &middot; ATR</text>
    <line x1="120" y1="300" x2="700" y2="300" stroke="#b42318" stroke-width="2"/>
    <text x="128" y="319" fill="#b42318" font-size="13" font-weight="600">stop barrier  &minus;sl &middot; ATR</text>
    <line x1="700" y1="60" x2="700" y2="320" stroke="#6b7680" stroke-width="1.6" stroke-dasharray="5 4"/>
    <text x="700" y="352" fill="#6b7680" font-size="13" font-weight="600" text-anchor="middle">time cap</text>
    <line x1="120" y1="210" x2="700" y2="210" stroke="#e3e8ea" stroke-width="1" stroke-dasharray="2 4"/>
    <text x="128" y="204" fill="#8a939a" font-size="11">entry price</text>
    <line x1="98" y1="210" x2="98" y2="300" stroke="#8a939a" stroke-width="1"/>
    <line x1="94" y1="210" x2="102" y2="210" stroke="#8a939a" stroke-width="1"/>
    <line x1="94" y1="300" x2="102" y2="300" stroke="#8a939a" stroke-width="1"/>
    <text x="90" y="258" fill="#6b7680" font-size="12" font-weight="600" text-anchor="end" transform="rotate(-90 90 258)">1R = entry&rarr;stop</text>
    <polyline fill="none" stroke="#37474f" stroke-width="2.2"
      points="120,210 158,224 196,198 234,214 272,182 310,196 348,158 386,172 424,138 462,150 500,120 538,132 576,100 606,88 628,80"/>
    <circle cx="120" cy="210" r="5" fill="#00695c"/>
    <text x="120" y="188" fill="#00695c" font-size="12.5" font-weight="600" text-anchor="middle">entry (signal bar)</text>
    <circle cx="628" cy="80" r="6" fill="#1a7f37" stroke="#ffffff" stroke-width="1.5"/>
    <text x="636" y="70" fill="#1a7f37" font-size="12.5" font-weight="600">first touch &rarr; win, label +1</text>
    <text x="445" y="392" fill="#38454c" font-size="12.5" text-anchor="middle">R = profit in units of 1R &middot; profit barrier first &rarr; +2.5R here (tp/sl = 2.5/1.0); stop first &rarr; &minus;1R; neither &rarr; timed-out remainder</text>
  </svg>
</div></body></html>"""


def _sheet(inner: str, width: int, css: str) -> str:
    """Wrap report-block HTML in a light-forced page for headless capture."""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<style>body{{margin:0;background:var(--cr-bg);}}{css}</style></head>'
            f'<body data-theme="light"><div class="cr-wrap" style="max-width:{width}px;'
            f'padding:20px">{inner}</div></body></html>')


def report_sheets() -> dict[str, str]:
    """Build the three report figure sheets from the §12 Donchian gauntlet."""
    spec = importlib.util.spec_from_file_location("dg", REPO / "examples" / "donchian_gauntlet.py")
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)
    import numpy as np
    from crucible.edge import reality_check, expectancy
    from crucible.validation import walk_forward, run_gauntlet, Thresholds
    from crucible.report import (verdict_banner, metrics_table, verdict_summary,
                                 gate_block, cumulative_r, report_css)
    from crucible.report.tearsheet import _PILLARS, _plotly, _COSTS_NOT_ATTESTED

    px = dg.synthetic_prices()
    tp, sl, to = 2.5, 1.0, 30
    wf = walk_forward(px, dg.donchian, param_grid={"lookback": [20, 40]},
                      is_days=365 * 3, oos_days=365, tp=tp, sl=sl, timeout=to)
    g = run_gauntlet(wf.stitched, prices=px, wf=wf, side="long", tp=tp, sl=sl,
                     n_variants=2, thr=Thresholds(n_boot=5000, n_perm=5000, n_random_sims=500))
    trades = wf.stitched
    css = report_css()
    # the §12 synthetic book carries no costs, so the report's default cost note is
    # the honest read here — mirror what gauntlet_report() stamps under the verdict
    note = f"<div class='cr-hostnote'>{_COSTS_NOT_ATTESTED}</div>"

    hero = (verdict_banner(g, title="Donchian breakout — the gauntlet",
                           subtitle="stitched OOS log · 20/40 lookback search")
            + metrics_table(trades) + verdict_summary(g) + note)
    by = {x.name: x for x in g.gates}
    gates = "".join(gate_block(by[p]) for p in _PILLARS if p in by)

    # Amber "scope-limited" illustration (#41): real strong metrics paired with a
    # synthetic gauntlet whose only miss is GENERAL (core REAL/STRONG/DURABLE hold),
    # so the report renders its third verdict state — EDGE VALIDATED · scope-limited.
    @dataclass
    class _FakeGate:
        name: str
        passed: bool

    class _FakeGauntlet:
        def __init__(self, gates):
            self.gates = gates

        @property
        def passed(self):
            return all(x.passed for x in self.gates)

    scope_g = _FakeGauntlet([_FakeGate("REAL", True), _FakeGate("STRONG", True),
                             _FakeGate("DURABLE", True), _FakeGate("GENERAL", False)])
    scope = (verdict_banner(scope_g, title="A scope-limited edge",
                            subtitle="core validated · cross-market generalization unproven")
             + metrics_table(trades) + verdict_summary(scope_g) + note)

    go, _ = _plotly()
    cr = cumulative_r(trades)
    grid = "rgba(128,128,128,0.18)"
    fig = go.Figure(go.Scatter(x=list(cr.index), y=cr.values, mode="lines",
                               line=dict(color="#1a7f37", width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.update_layout(height=340, showlegend=False, margin=dict(t=44, l=48, r=24, b=36),
                      title="Cumulative R (stitched out-of-sample)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#5b6570"))
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    chart = fig.to_html(full_html=False, include_plotlyjs=True)

    # Bootstrap-expectancy panel (the §3 picture): the distribution of expectancy
    # over resamples, with the 95% CI (amber) and point (green) against zero.
    r = trades.r
    v = reality_check(trades, n_boot=10_000, seed=0)
    rng = np.random.default_rng(0)
    boot = np.array([expectancy(rng.choice(r, size=len(r), replace=True)) for _ in range(5000)])
    figb = go.Figure(go.Histogram(x=boot, nbinsx=40, marker_color="#b0b0b0"))
    for xval, dash, color in ((0, "dot", "#888"), (v.ci.low, "dash", "#9a6700"),
                              (v.ci.high, "dash", "#9a6700"), (v.point, "solid", "#1a7f37")):
        figb.add_vline(x=xval, line_dash=dash, line_color=color)
    figb.update_layout(height=340, showlegend=False, margin=dict(t=44, l=48, r=24, b=40),
                       title="Bootstrap expectancy — 95% CI (amber) vs the point estimate (green)",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#5b6570"))
    figb.update_xaxes(gridcolor=grid, zerolinecolor=grid, title_text="expectancy (R)")
    figb.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    chartb = figb.to_html(full_html=False, include_plotlyjs=True)

    return {
        "gauntlet_hero": _sheet(hero, 840, css),
        "gauntlet_scope": _sheet(scope, 840, css),
        "gauntlet_bootstrap": _sheet(chartb, 840, css),
        "gauntlet_gates": _sheet(gates, 840, css),
        "gauntlet_cumr": _sheet(chart, 840, css),
    }


PURGE_EMBARGO_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
  body{margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  .wrap{max-width:820px;margin:0 auto;padding:22px}
  h2{margin:0 0 2px;font-size:15px;color:#00695c}
  .sub{font-size:12.5px;color:#6b7680;margin-bottom:8px;line-height:1.45}
  text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
</style></head><body><div class="wrap">
  <h2>Purge &amp; embargo &mdash; making the train/test split leakage-proof</h2>
  <div class="sub">A trade whose forward window straddles the split can't sit in train (it would peek past the boundary); the embargo drops the first slice of test to kill autocorrelation across the seam.</div>
  <svg viewBox="0 0 820 300" width="100%" xmlns="http://www.w3.org/2000/svg">
    <text x="278" y="24" fill="#00695c" font-size="13" font-weight="700" text-anchor="middle">TRAIN &mdash; fit / choose here</text>
    <text x="648" y="24" fill="#004d40" font-size="13" font-weight="700" text-anchor="middle">TEST &mdash; the untouched verdict</text>
    <rect x="470" y="40" width="70" height="212" fill="#f0c674" fill-opacity="0.22"/>
    <text x="505" y="272" fill="#9a6700" font-size="11.5" font-weight="700" text-anchor="middle">embargo</text>
    <text x="505" y="286" fill="#9a6700" font-size="11" text-anchor="middle">(dropped)</text>
    <line x1="470" y1="36" x2="470" y2="252" stroke="#6b7680" stroke-width="1.6" stroke-dasharray="5 4"/>
    <text x="470" y="49" fill="#6b7680" font-size="12" font-weight="700" text-anchor="middle">split</text>
    <line x1="90" y1="252" x2="792" y2="252" stroke="#c7ced2" stroke-width="1"/>
    <text x="786" y="270" fill="#8a939a" font-size="11.5" text-anchor="end">time &rarr;</text>
    <g stroke="#00695c" stroke-width="3.4" stroke-linecap="round"><line x1="120" y1="78" x2="214" y2="78"/><line x1="238" y1="106" x2="352" y2="106"/><line x1="372" y1="134" x2="452" y2="134"/></g>
    <g fill="#00695c"><circle cx="120" cy="78" r="4.2"/><circle cx="214" cy="78" r="4.2"/><circle cx="238" cy="106" r="4.2"/><circle cx="352" cy="106" r="4.2"/><circle cx="372" cy="134" r="4.2"/><circle cx="452" cy="134" r="4.2"/></g>
    <text x="120" y="70" fill="#38454c" font-size="11">train trades &mdash; enter and exit before the split</text>
    <line x1="430" y1="176" x2="528" y2="176" stroke="#b0868a" stroke-width="3" stroke-linecap="round" stroke-dasharray="6 4"/>
    <circle cx="430" cy="176" r="4.2" fill="#b0868a"/><circle cx="528" cy="176" r="4.2" fill="#b0868a"/>
    <text x="430" y="168" fill="#8a3d34" font-size="11" font-weight="600">purged &mdash; straddles the split, so it can't be in train</text>
    <g stroke="#185fa5" stroke-width="3.4" stroke-linecap="round"><line x1="566" y1="204" x2="648" y2="204"/><line x1="676" y1="232" x2="760" y2="232"/></g>
    <g fill="#185fa5"><circle cx="566" cy="204" r="4.2"/><circle cx="648" cy="204" r="4.2"/><circle cx="676" cy="232" r="4.2"/><circle cx="760" cy="232" r="4.2"/></g>
    <text x="566" y="196" fill="#38454c" font-size="11">test trades &mdash; begin only after the embargo</text>
  </svg>
</div></body></html>"""


def ml_sheets() -> dict[str, str]:
    """Build the two §13 figures from examples/ml_meta_label.py: the score's
    quantile-decay bars and the unfiltered-vs-filtered gauntlet verdicts."""
    spec = importlib.util.spec_from_file_location("mm", REPO / "examples" / "ml_meta_label.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    from crucible.ml import quantile_decay
    from crucible.edge import TradeLog
    from crucible.validation import run_gauntlet
    from crucible.report import verdict_banner, report_css
    from crucible.report.tearsheet import _plotly

    go, _ = _plotly()
    css = report_css()
    book = mm.synthetic_meta_book()
    grid = "rgba(128,128,128,0.18)"

    t = quantile_decay(book[["score", "label"]], q=5).table
    fig = go.Figure(go.Bar(x=[f"Q{int(q)}" for q in t["quantile"]], y=t["win_rate"].to_numpy(),
                           text=[f"{v*100:.1f}%" for v in t["win_rate"]], textposition="outside",
                           marker_color="#1d9e75"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#9a6700",
                  annotation_text="coin flip (50%)", annotation_position="top left")
    fig.update_layout(height=360, showlegend=False, margin=dict(t=48, l=54, r=24, b=36),
                      title="Win rate by score quantile (worst → best)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#5b6570"))
    fig.update_yaxes(tickformat=".0%", gridcolor=grid, zerolinecolor=grid, title="realized win rate")
    fig.update_xaxes(gridcolor=grid)
    decay = fig.to_html(full_html=False, include_plotlyjs=True)

    take = book[book["score"] >= book["score"].quantile(0.60)]
    g_all = run_gauntlet(TradeLog.from_arrays(r=book["r"].to_numpy()), n_variants=1)
    g_take = run_gauntlet(TradeLog.from_arrays(r=take["r"].to_numpy()), n_variants=1)
    lab = "font:600 12px/1.4 -apple-system,system-ui,sans-serif;color:#6b7680;margin:0 0 4px"
    verdict = (f"<div style='{lab}'>Primary book — 1,600 candidate trades</div>"
               + verdict_banner(g_all, title="Unfiltered", subtitle="every candidate trade")
               + f"<div style='{lab};margin-top:22px'>Filtered — top 40% by score, 640 trades</div>"
               + verdict_banner(g_take, title="Filtered", subtitle="the model's take/skip filter"))

    return {"ml_decay": _sheet(decay, 720, css), "ml_verdict": _sheet(verdict, 720, css)}


def render_png(html: str, out: Path, pad: int = 22) -> None:
    """Headless-render HTML to a 2x PNG, then autocrop the page background."""
    from PIL import Image, ImageChops
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(html)
        src = fh.name
    raw = out.with_suffix(".raw.png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", "--window-size=880,1500",
                    f"--screenshot={raw}", f"file://{src}"],
                   check=True, capture_output=True)
    im = Image.open(raw).convert("RGB")
    bg = im.getpixel((2, 2))
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, bg)).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    im.save(out)
    raw.unlink(missing_ok=True)
    print(f"wrote {out.relative_to(REPO)}  ({im.width}x{im.height})")


# Two committed colorways (a PNG gets no page context to drive currentColor):
#   full — slate vessel that reads on light and dark (title area, PDF cover, favicon)
#   mono — all-white for the teal site header bar
LOGO_FULL = dict(vessel="#7a808a", molten="#e0812b", up="#1a7f37", down="#b42318")
LOGO_MONO = dict(vessel="#ffffff", molten="#ffffff", up="#ffffff", down="#ffffff")


def render_logo(out: Path, colors: dict = LOGO_FULL, size: int = 96, pad: int = 6) -> None:
    """Render the crucible mark to a transparent, alpha-cropped PNG in the given
    colorway (see LOGO_FULL / LOGO_MONO)."""
    from PIL import Image
    from crucible.report.tearsheet import _logo_svg
    svg = _logo_svg(size=size, **colors)
    html = f'<!doctype html><html><head><meta charset="utf-8">' \
           f'<style>html,body{{margin:0;background:transparent}}</style></head>' \
           f'<body>{svg}</body></html>'
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(html)
        src = fh.name
    raw = out.with_suffix(".raw.png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", "--default-background-color=00000000",
                    f"--window-size={size + 20},{size + 20}",
                    f"--screenshot={raw}", f"file://{src}"],
                   check=True, capture_output=True)
    im = Image.open(raw).convert("RGBA")
    bbox = im.split()[-1].getbbox()  # crop to the alpha (non-transparent) bounds
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    im.save(out)
    raw.unlink(missing_ok=True)
    print(f"wrote {out.relative_to(REPO)}  ({im.width}x{im.height}, transparent)")


def main() -> None:
    IMG.mkdir(exist_ok=True)
    sheets = dict(report_sheets())
    sheets.update(ml_sheets())
    sheets["triple_barrier"] = TRIPLE_BARRIER_HTML
    sheets["gate_ladder"] = GATE_LADDER_HTML
    sheets["meta_label"] = META_LABEL_HTML
    sheets["purge_embargo"] = PURGE_EMBARGO_HTML
    for name, html in sheets.items():
        render_png(html, IMG / f"{name}.png")
    render_logo(IMG / "crucible_logo.png", LOGO_FULL)
    render_logo(IMG / "crucible_logo_white.png", LOGO_MONO)  # site header (teal bar)


if __name__ == "__main__":
    main()
