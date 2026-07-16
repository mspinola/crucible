import pytest

pytest.importorskip("plotly")  # report is behind the [report] extra

from crucible.edge import barrier_trades          # noqa: E402
from crucible.strategies import ma_cross           # noqa: E402
from crucible.report import (                      # noqa: E402
    tearsheet, cumulative_r, gauntlet_report, verdict_banner, verdict_summary,
    gate_block, edge_panels, metrics_table, report_css,
)
from crucible.validation import run_gauntlet, walk_forward  # noqa: E402


def _full_gauntlet(ohlc):
    """A gauntlet with all four pillars run: REAL (prices → null), STRONG,
    DURABLE (walk-forward), GENERAL (cross-market)."""
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long", tp=2.0, sl=1.0, timeout=20)
    wf = walk_forward(ohlc, ma_cross, {"fast": [10, 20], "slow": [50, 100]}, side="long")
    others = barrier_trades(ohlc, ma_cross(ohlc, fast=15, slow=60), side="long")
    return tl, run_gauntlet(tl, prices=ohlc, wf=wf,
                            trade_logs={"MAIN": tl, "ALT": others})


def test_cumulative_r_is_monotone_sum(ohlc):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long")
    cr = cumulative_r(tl)
    assert len(cr) == tl.n
    assert cr.iloc[-1] == pytest.approx(tl.r.sum())


def test_tearsheet_writes_self_contained_html(ohlc, tmp_path):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long", tp=2.0, sl=1.0, timeout=20)
    out = tmp_path / "sheet.html"
    path = tearsheet(tl, str(out), title="test sheet", subtitle="synthetic")
    assert path == str(out) and out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<title>test sheet</title>" in html
    assert any(v in html for v in ("HELD", "FRAGILE", "FAIL"))   # verdict rendered
    assert "plotly" in html.lower()                              # js inlined
    assert out.stat().st_size > 100_000                          # self-contained (plotly.js)


def test_tearsheet_handles_empty_log(tmp_path):
    from crucible.edge import TradeLog
    tl = TradeLog.from_arrays(r=[], entry_date=[], exit_date=[])
    out = tmp_path / "empty.html"
    tearsheet(tl, str(out))
    assert out.exists()


# --------------------------------------------------------------------------- #
# Composable blocks
# --------------------------------------------------------------------------- #
def test_metrics_table_renders_core_rows(ohlc):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long")
    html = metrics_table(tl)
    for label in ("Trades", "Win rate", "Expectancy", "Profit factor", "SQN-100"):
        assert label in html
    assert "cr-metrics" in html
    # horizontal strip: one cell per metric, not a tall <table>
    assert "<table" not in html
    assert html.count("cr-metric'") >= 6      # one flex cell per metric


def test_verdict_summary_pass_and_fail(ohlc):
    tl, g = _full_gauntlet(ohlc)
    s = verdict_summary(g)
    assert "cr-summary" in s
    assert ("Validated" in s) or ("Not validated" in s)
    # the redundant "(N/N pillars — X fails)" count is deduped — the banner + chips
    # carry it; the summary keeps only the plain-English reading
    assert "pillars" not in s
    # a REAL-only + STRONG-only gauntlet still summarizes without the unrun pillars
    from crucible.validation import run_gauntlet
    g2 = run_gauntlet(barrier_trades(ohlc, ma_cross(ohlc), side="long"))
    s2 = verdict_summary(g2)
    assert "DURABLE" not in s2 and "GENERAL" not in s2   # speaks only to pillars that ran


def test_metrics_table_skips_nan_excursion():
    from crucible.edge import TradeLog
    import numpy as np
    # A rules book carries NaN mfe/mae (excursion is not defined for it) — the
    # excursion / exit-efficiency rows must be omitted, never rendered as "nan".
    tl = TradeLog.from_arrays(r=np.random.default_rng(0).normal(0.1, 1, 60),
                              mfe=np.full(60, np.nan), mae=np.full(60, np.nan))
    html = metrics_table(tl)
    assert "nan" not in html.lower()
    assert "Excursion ratio" not in html and "Exit efficiency" not in html


def test_edge_panels_omit_plotlyjs_by_default(ohlc):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long")
    frag = edge_panels(tl, include_plotlyjs=False)
    assert "plotly-graph-div" in frag or "class=\"plotly" in frag
    # Default omits the ~3.5MB library so the host page can load it once.
    assert len(frag) < 200_000
    inlined = edge_panels(tl, include_plotlyjs=True)
    assert len(inlined) > len(frag)


def test_gate_block_shows_verdict_and_checks(ohlc):
    _, g = _full_gauntlet(ohlc)
    by = {gate.name: gate for gate in g.gates}
    html = gate_block(by["REAL"])
    assert "Real" in html                       # pillar heading (title-cased)
    assert "cr-tag" in html                      # PASS/FAIL badge
    assert ("pass" in html) or ("fail" in html)
    # every check name is rendered
    for c in by["REAL"].checks:
        assert c.name in html


def test_gate_block_collapse_follows_verdict(ohlc):
    _, g = _full_gauntlet(ohlc)
    for gate in g.gates:
        html = gate_block(gate)
        assert html.startswith("<details")
        opened = "cr-gate' open>" in html
        assert opened == (not gate.passed)      # failed → expanded, passed → collapsed
        # even collapsed, the summary carries name + badge (nothing important hidden)
        assert "gate-h" in html and "cr-tag" in html
    # explicit override wins in both directions
    any_gate = g.gates[0]
    assert "cr-gate' open>" in gate_block(any_gate, expanded=True)
    assert "cr-gate' open>" not in gate_block(any_gate, expanded=False)


def test_verdict_banner_lists_all_four_pillars(ohlc):
    _, g = _full_gauntlet(ohlc)
    banner = verdict_banner(g, title="book", subtitle="synthetic")
    assert "GAUNTLET" in banner
    for p in ("REAL", "STRONG", "DURABLE", "GENERAL"):
        assert p in banner


def test_verdict_banner_marks_unrun_pillars(ohlc):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long")
    g = run_gauntlet(tl)                          # REAL + STRONG only
    banner = verdict_banner(g)
    # DURABLE / GENERAL did not run → shown as not-applicable, not pass/fail
    assert "na" in banner and "DURABLE" in banner and "GENERAL" in banner


def test_verdict_banner_relabels_unrun_pillar(ohlc):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long")
    g = run_gauntlet(tl)                          # GENERAL did not run
    banner = verdict_banner(g, pillar_notes={"GENERAL": "&rarr; strategy report"})
    assert "GENERAL &rarr; strategy report" in banner   # relabelled, no bare dash
    # a pillar without a note still falls back to the em dash
    assert "DURABLE —" in banner


def test_gauntlet_report_embeds_all_pillars_and_appendix(ohlc, tmp_path):
    tl, g = _full_gauntlet(ohlc)
    out = tmp_path / "gauntlet.html"
    appendix = "<section id='npf-appendix'>capital-aware stuff</section>"
    path = gauntlet_report(g, tl, str(out), title="npf book",
                           subtitle="deployed", appendix_html=appendix)
    assert path == str(out) and out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<title>npf book</title>" in html
    assert "GAUNTLET" in html
    for p in ("Real", "Strong", "Durable", "General"):
        assert p in html                          # each pillar section rendered
    assert "npf-appendix" in html                 # host appendix embedded
    assert "plotly" in html.lower()               # edge panels present


def test_gauntlet_report_returns_string_without_path(ohlc):
    tl, g = _full_gauntlet(ohlc)
    doc = gauntlet_report(g, tl, include_plotlyjs=False)
    assert doc.startswith("<!doctype html>") and "GAUNTLET" in doc


def test_gauntlet_report_embeds_header_html(ohlc):
    tl, g = _full_gauntlet(ohlc)
    doc = gauntlet_report(g, tl, include_plotlyjs=False,
                          header_html="<div id='net-gross-badge'>NET</div>")
    assert "net-gross-badge" in doc
    # header sits under the verdict banner, before the pillar sections
    assert doc.index("net-gross-badge") < doc.index("Real")


def test_report_css_is_style_body_only():
    css = report_css()
    assert "<style>" not in css and ".cr-gate" in css
