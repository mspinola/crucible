import pytest

pytest.importorskip("plotly")  # report is behind the [report] extra

from crucible.edge import barrier_trades          # noqa: E402
from crucible.strategies import ma_cross           # noqa: E402
from crucible.report import tearsheet, cumulative_r  # noqa: E402


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
