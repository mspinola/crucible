import pytest

pytest.importorskip("plotly")  # report is behind the [report] extra

from crucible.edge import barrier_trades          # noqa: E402
from crucible.strategies import ma_cross           # noqa: E402
from crucible.report import fullrange_scorecard, holdout_report  # noqa: E402


def test_fullrange_scorecard_is_a_scorecard_not_the_gauntlet(ohlc, tmp_path):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long", tp=2.0, sl=1.0, timeout=20)
    out = tmp_path / "fr.html"
    path = fullrange_scorecard(tl, str(out), title="fr test", subtitle="synthetic")
    assert path == str(out) and out.exists()
    html = out.read_text(encoding="utf-8")

    assert "<title>fr test</title>" in html
    assert any(v in html for v in ("HELD", "FRAGILE", "FAIL"))     # verdict rendered
    # scorecard chrome: the eyebrow + stat tiles (verdict / total R / R-per-year)
    assert "cr-eyebrow" in html and "cr-tiles" in html and "cr-verdict-tile" in html
    assert "total R" in html and "R / year" in html
    # NOT the gauntlet report: none of its verdict-banner labels are rendered
    assert not any(s in html for s in ("GAUNTLET PASS", "GAUNTLET FAIL", "EDGE VALIDATED"))
    assert "plotly" in html.lower() and out.stat().st_size > 100_000
    assert "cr-title" in html and "aria-label='crucible'" in html


def test_fullrange_scorecard_drops_r_per_year_without_dates(tmp_path):
    from crucible.edge import TradeLog
    tl = TradeLog.from_arrays(r=[1.0, -1.0, 2.0, -1.0, 1.5])   # no entry/exit dates
    html = fullrange_scorecard(tl)     # path=None -> returns the HTML
    assert "total R" in html          # total R still shown
    assert "R / year" not in html     # no dates -> no per-year tile


def test_holdout_report_renders_train_and_emphasized_test(ohlc, tmp_path):
    tl = barrier_trades(ohlc, ma_cross(ohlc), side="long", tp=2.0, sl=1.0, timeout=20)
    out = tmp_path / "ho.html"
    path = holdout_report(tl, "2018-01-01", str(out), embargo_weeks=2)
    assert path == str(out) and out.exists()
    html = out.read_text(encoding="utf-8")

    assert "<title>Holdout scorecard</title>" in html
    assert "cr-split" in html and "cr-honest" in html             # two cards, test emphasized
    assert "Train" in html and "Test" in html
    assert "the honest read" in html                              # test is the verdict
    assert "split 2018-01-01" in html and "embargo 2w" in html
    assert "plotly" in html.lower()
