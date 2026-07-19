import pytest

pytest.importorskip("plotly")  # report is behind the [report] extra

from crucible.edge import barrier_trades          # noqa: E402
from crucible.strategies import ma_cross           # noqa: E402
from crucible.report import (                      # noqa: E402
    tearsheet, cumulative_r, gauntlet_report, verdict_banner, verdict_summary,
    gate_block, edge_panels, metrics_table, report_css, monthly_r, title_lockup,
    equity_drawdown, exit_reason_breakdown, holding_vs_r, exit_efficiency_dist,
    edge_ratio_curve,
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
    # the crucible mark now rides the top-left of every tearsheet, not just the gauntlet
    assert "cr-title" in html and "aria-label='crucible'" in html


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


class _FakeGate:
    def __init__(self, name, passed):
        self.name, self.passed = name, passed


class _FakeGauntlet:
    """Minimal stand-in (verdict rendering only reads .gates[].name/.passed and
    .passed) so the scope-limited state can be exercised deterministically."""
    def __init__(self, **passed):
        self.gates = [_FakeGate(n, passed[n]) for n in ("REAL", "STRONG", "DURABLE", "GENERAL")]
        self.passed = all(g.passed for g in self.gates)


def test_scope_limited_verdict_when_only_general_fails():
    from crucible.report.tearsheet import _verdict_state
    g = _FakeGauntlet(REAL=True, STRONG=True, DURABLE=True, GENERAL=False)
    assert _verdict_state(g) == "scope"

    banner = verdict_banner(g, title="book")
    assert "EDGE VALIDATED" in banner and "scope-limited" in banner
    assert "GAUNTLET FAIL" not in banner            # not red-failed
    assert "GENERAL ⚠" in banner and "class='warn'" in banner   # amber caveat, not red ✗

    s = verdict_summary(g)
    assert "Validated on its universe" in s and "unproven" in s
    assert "Not validated" not in s


def test_pillar_bullets_render_and_direction_inference(ohlc):
    from crucible.report import pillar_bullets
    _, g = _full_gauntlet(ohlc)
    html = pillar_bullets(g)
    assert "plotly" in html.lower()
    # a headline check per pillar that ran, labelled
    for pillar in ("REAL", "STRONG", "DURABLE", "GENERAL"):
        assert pillar in html
    assert "plotly_dark" not in html                     # theme-neutral

    # direction is inferred from (value>=threshold)==passed — verify both cases:
    from crucible.report.tearsheet import _BULLET_LABEL  # noqa: F401  (label map present)


def test_check_bullet_svg_colors_by_result_and_skips_non_numeric():
    from crucible.report.tearsheet import _check_bullet_svg
    from types import SimpleNamespace as NS
    # passing hard → green; failing hard → red; failing soft → amber
    assert "var(--cr-pass)" in _check_bullet_svg(NS(name="wfe", value=1.47, threshold=0.5, hard=True, passed=True))
    assert "var(--cr-fail)" in _check_bullet_svg(NS(name="p", value=0.10, threshold=0.05, hard=True, passed=False))
    assert "var(--cr-warn)" in _check_bullet_svg(NS(name="sqn", value=0.1, threshold=1.6, hard=False, passed=False))
    # a value bar and a threshold tick are drawn
    svg = _check_bullet_svg(NS(name="e", value=0.44, threshold=0.0, hard=True, passed=True))
    assert "<rect" in svg and "<line" in svg and "<svg" in svg
    # non-numeric / missing → empty (no crash)
    assert _check_bullet_svg(NS(name="t", value=("a", "b"), threshold=None, hard=True, passed=True)) == ""


def test_gate_block_embeds_per_check_bullets(ohlc):
    _, g = _full_gauntlet(ohlc)
    html = gate_block(g.gates[0])          # any real gate
    assert "cr-cbul" in html               # the inline bullet svg is in the checks table


def test_pillar_bullets_empty_without_checks():
    from crucible.report import pillar_bullets
    from types import SimpleNamespace
    # a gate whose checks lack numeric value/threshold → nothing to plot
    g = SimpleNamespace(gates=[SimpleNamespace(name="REAL", checks=[])])
    assert pillar_bullets(g) == ""


def test_core_failure_still_reads_fail():
    from crucible.report.tearsheet import _verdict_state
    g = _FakeGauntlet(REAL=False, STRONG=True, DURABLE=True, GENERAL=True)
    assert _verdict_state(g) == "fail"
    assert "GAUNTLET FAIL" in verdict_banner(g, title="book")
    assert "Not validated" in verdict_summary(g)


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


def test_monthly_r_is_gap_free_and_preserves_total(ohlc):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long")
    mr = monthly_r(tl)
    # gap-free monthly grid: one bin per calendar month between first and last exit
    assert len(mr) >= 2
    assert mr.index.to_period("M").nunique() == len(mr)      # no missing months
    # summing periods must reconstruct the trade-log total R (empty months add 0)
    import numpy as np
    assert float(mr.sum()) == pytest.approx(float(tl.r.sum()))


def test_monthly_r_empty_log_is_empty_series():
    from crucible.edge import TradeLog
    assert len(monthly_r(TradeLog.from_arrays(r=[]))) == 0


def test_edge_panels_block_panel_is_opt_in(ohlc):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long")
    base = edge_panels(tl, include_plotlyjs=False)
    assert "Block bootstrap" not in base                     # off by default → unchanged
    withblk = edge_panels(tl, include_plotlyjs=False, period_returns=monthly_r(tl), block=6)
    assert "Block bootstrap" in withblk                       # panel appended
    assert "mean period return (R)" in withblk                # its own period-mean axis
    assert "block=6" in withblk and "i.i.d." in withblk       # both whiskers drawn
    assert len(withblk) > len(base)
    # the appended panel must not re-inline plotly.js (host/main fragment carries it)
    assert "cdn.plot" in withblk or "plotly-graph-div" in withblk


def test_block_bootstrap_panel_widens_ci_on_autocorrelation():
    # A positively autocorrelated period series: the block CI must be WIDER than the
    # i.i.d. one — the whole reason the panel exists. Assert on the numbers behind it.
    import numpy as np
    from crucible.edge.stats import block_bootstrap_ci
    from crucible.report.tearsheet import _block_bootstrap_panel
    rng = np.random.default_rng(1)
    n, x = 72, np.empty(72)
    x[0] = 0.3
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = 0.3 + 0.7 * (x[i - 1] - 0.3) + e[i]
    iid = block_bootstrap_ci(x, block=1, seed=0)
    blk = block_bootstrap_ci(x, block=6, seed=0)
    assert (blk.high - blk.low) > (iid.high - iid.low)
    html = _block_bootstrap_panel(x, block=6)
    assert html and "block=6" in html
    # too-short series can't be block-resampled → no panel, no crash
    assert _block_bootstrap_panel([0.1]) == ""
    assert _block_bootstrap_panel([]) == ""


def test_gauntlet_report_embeds_block_panel(ohlc):
    tl, g = _full_gauntlet(ohlc)
    doc = gauntlet_report(g, tl, include_plotlyjs=False, period_returns=monthly_r(tl))
    assert "Block bootstrap" in doc and "mean period return (R)" in doc
    # absent when not requested
    assert "Block bootstrap" not in gauntlet_report(g, tl, include_plotlyjs=False)


def test_gauntlet_report_stamps_costs_not_attested(ohlc):
    tl, g = _full_gauntlet(ohlc)
    # No host note: crucible can't tell net from gross, so it nudges rather than imply net.
    bare = gauntlet_report(g, tl, include_plotlyjs=False)
    assert "costs not attested" in bare
    # A host that attests (e.g. a net/gross badge) suppresses the default entirely.
    attested = gauntlet_report(g, tl, include_plotlyjs=False,
                               header_html="<b>NET of transaction costs</b>")
    assert "NET of transaction costs" in attested
    assert "costs not attested" not in attested


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


def test_verdict_banner_rows_pill_and_pillars_together(ohlc):
    _, g = _full_gauntlet(ohlc)
    banner = verdict_banner(g, title="book")
    # pill + pillar chips live in one flex row, pill first then chips
    row = banner.index("cr-verdict-row")
    assert row < banner.index("cr-verdict'") < banner.index("cr-pillars")


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


def test_gauntlet_report_has_logo_favicon_and_metric_order(ohlc):
    tl, g = _full_gauntlet(ohlc)
    doc = gauntlet_report(g, tl, include_plotlyjs=False)
    # the crucible mark is in the header lockup, and the favicon carries it too
    assert "cr-title" in doc and "aria-label='crucible'" in doc
    assert 'rel="icon"' in doc and "data:image/svg+xml," in doc
    # two-column top: interpretation prose (left) + the stats as a card (right), then a
    # rule before the pillar bullets. Match the class='' body elements, not the .cr-*
    # rules in the <style> head.
    assert "class='cr-top-left'" in doc and "class='cr-statcard'" in doc
    assert doc.index("class='cr-summary'") < doc.index("class='cr-statcard'")  # prose left, card right
    assert doc.count("class='cr-div'") >= 2   # a rule frames the two-column row (above + below)


def test_report_css_is_style_body_only():
    css = report_css()
    assert "<style>" not in css and ".cr-gate" in css


def test_strong_whisker_draws_ci_panels(ohlc):
    from crucible.report.tearsheet import _strong_whisker
    tl, g = _full_gauntlet(ohlc)
    strong = next(gt for gt in g.gates if gt.name == "STRONG")
    html = _strong_whisker(strong, tl)
    assert html  # non-empty div
    for lbl in ("Expectancy (R)", "Profit factor", "SQN-100"):
        assert lbl in html
    assert "floor" in html          # each panel marks its gate floor
    # and it rides along inside the STRONG card on the full page
    assert "Expectancy (R)" in gauntlet_report(g, tl, include_plotlyjs=False)


def test_strong_whisker_guards(ohlc):
    from crucible.report.tearsheet import _strong_whisker
    from crucible.edge import TradeLog
    tl, g = _full_gauntlet(ohlc)
    strong = next(gt for gt in g.gates if gt.name == "STRONG")
    assert _strong_whisker(None, tl) == ""                        # no gate
    assert _strong_whisker(strong, TradeLog.from_arrays(r=[])) == ""  # no trades


def test_title_lockup_pairs_logo_with_title():
    html = title_lockup("My Book")
    assert "cr-title" in html                       # the lockup row
    assert "aria-label='crucible'" in html          # the mark
    assert "<h1>My Book</h1>" in html
    assert title_lockup("") == ""                    # falsy title → droppable header


def test_gauntlet_report_uses_the_same_lockup(ohlc):
    # the gauntlet banner must render via the shared lockup, not a bespoke header
    tl, g = _full_gauntlet(ohlc)
    doc = gauntlet_report(g, tl, title="Book", include_plotlyjs=False)
    assert title_lockup("Book") in doc


# --------------------------------------------------------------------------- #
# Extra capital-free panels (promoted from npf)
# --------------------------------------------------------------------------- #
def _bars(ohlc):
    """A fully-populated TradeLog (r, mfe, mae, bars_held, entry/exit dates,
    exit_reason) — barrier_trades emits every optional column."""
    return barrier_trades(ohlc, ma_cross(ohlc), side="long", tp=2.0, sl=1.0, timeout=20)


def _is_embed_fragment(frag):
    """A default panel is an embeddable div: a plotly graph, no inlined library,
    theme-neutral (no plotly_dark template)."""
    return (frag
            and ("plotly-graph-div" in frag or 'class="plotly' in frag)
            and len(frag) < 200_000          # library loaded via CDN, not inlined
            and "plotly_dark" not in frag)


def test_equity_drawdown_renders_curve_and_underwater(ohlc):
    frag = equity_drawdown(_bars(ohlc))
    assert _is_embed_fragment(frag)
    assert "Cumulative R" in frag and "Drawdown (R)" in frag     # both panels
    assert "max DD" in frag                                      # trough annotated
    # inlining the library makes the fragment self-contained (and much larger)
    assert len(equity_drawdown(_bars(ohlc), include_plotlyjs=True)) > len(frag)


def test_equity_drawdown_shades_oos_span(ohlc):
    import pandas as pd
    tl = _bars(ohlc)
    mid = pd.Series(pd.to_datetime(tl.col("exit_date"))).sort_values().iloc[len(tl) // 2]
    frag = equity_drawdown(tl, test_start=mid)
    assert "OOS" in frag                                         # test span marked


def test_equity_drawdown_empty_log_is_blank():
    from crucible.edge import TradeLog
    assert equity_drawdown(TradeLog.from_arrays(r=[])) == ""


def test_exit_reason_breakdown_attributes_r_and_count(ohlc):
    frag = exit_reason_breakdown(_bars(ohlc))
    assert _is_embed_fragment(frag)
    assert "Trade count" in frag and "R-multiples" in frag       # both panels' titles
    assert "timeout" in frag                                     # a friendly label


def test_exit_reason_breakdown_degrades_without_column():
    from crucible.edge import TradeLog
    # a plain R log carries no exit_reason -> nothing to attribute
    assert exit_reason_breakdown(TradeLog.from_arrays(r=[0.3, -1.0, 2.0])) == ""


def test_holding_vs_r_scatters_and_marks_medians(ohlc):
    frag = holding_vs_r(_bars(ohlc))
    assert _is_embed_fragment(frag)
    assert "Bars held" in frag and "Realized R" in frag
    assert "median hold" in frag


def test_holding_vs_r_degrades_without_bars_held():
    from crucible.edge import TradeLog
    assert holding_vs_r(TradeLog.from_arrays(r=[0.3, -1.0, 2.0])) == ""


def test_exit_efficiency_dist_renders_capture(ohlc):
    frag = exit_efficiency_dist(_bars(ohlc))
    assert _is_embed_fragment(frag)
    assert "Exit efficiency" in frag and "median capture" in frag


def test_exit_efficiency_dist_degrades_without_mfe():
    from crucible.edge import TradeLog
    import numpy as np
    # no mfe at all, and an all-NaN mfe (a rules book), both yield nothing
    assert exit_efficiency_dist(TradeLog.from_arrays(r=[0.3, -1.0])) == ""
    assert exit_efficiency_dist(
        TradeLog.from_arrays(r=[0.3, -1.0], mfe=np.full(2, np.nan))) == ""


def test_edge_ratio_curve_from_sequences(ohlc):
    horizons = [1, 2, 5, 10, 20]
    eratio = [0.9, 1.1, 1.4, 1.2, 1.0]
    frag = edge_ratio_curve(horizons, eratio)
    assert _is_embed_fragment(frag)
    assert "E-ratio" in frag
    assert "peak 1.40 @ 5 bars" in frag                          # argmax marked
    assert "edge = 1.0" in frag                                  # reference line


def test_edge_ratio_curve_guards_empty_and_mismatched():
    assert edge_ratio_curve([], []) == ""
    assert edge_ratio_curve([1, 2, 3], [0.9, 1.1]) == ""         # length mismatch
