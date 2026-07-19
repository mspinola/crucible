import numpy as np
import pandas as pd
import pytest

from crucible.edge import TradeLog
from crucible.edge.metrics import expectancy
from crucible.validation import windowed_segments
from crucible.validation.windows import OVERALL


def _log(rows):
    """rows: list of (r, entry_date, group)."""
    df = pd.DataFrame(rows, columns=["r", "entry_date", "grp"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return TradeLog(df)


def test_windows_align_to_start_year_and_span():
    tl = _log([(1.0, f"{y}-06-01", "A") for y in range(2012, 2021)])
    ws = windowed_segments(tl, by="grp", window_years=4, min_n=1)
    # edges aligned to the start year (2012), stepping by 4 to cover 2020
    assert ws.windows == [("2012", "2016"), ("2016", "2020"), ("2020", "2024")]


def test_windows_cell_counts_and_metric_are_correct():
    # two windows: 2012-2016 has 3 trades, 2016-2020 has 1
    tl = _log([
        (2.0, "2013-01-01", "A"), (-1.0, "2014-01-01", "A"), (2.0, "2015-01-01", "A"),
        (0.5, "2017-01-01", "A"),
    ])
    ws = windowed_segments(tl, by="grp", window_years=4, start="2012-01-01",
                           end="2018-01-01", min_n=1)
    a = ws.rows["A"]
    assert [c.n for c in a] == [3, 1]
    assert a[0].value == pytest.approx(expectancy(np.array([2.0, -1.0, 2.0])))
    assert a[1].value == pytest.approx(0.5)
    # OVERALL == A here (single group), same cells
    assert [c.n for c in ws.rows[OVERALL]] == [3, 1]


def test_windows_overall_pools_all_segments():
    tl = _log([
        (1.0, "2013-01-01", "A"), (1.0, "2013-02-01", "B"),
        (1.0, "2014-01-01", "A"), (-1.0, "2014-02-01", "B"),
    ])
    ws = windowed_segments(tl, by="grp", window_years=4, start="2012-01-01",
                           end="2015-01-01", min_n=1)
    # one window; OVERALL n == A n + B n
    assert len(ws.windows) == 1
    assert ws.rows[OVERALL][0].n == ws.rows["A"][0].n + ws.rows["B"][0].n == 4
    assert list(ws.rows) == [OVERALL, "A", "B"]      # OVERALL first, then sorted


def test_windows_min_n_flags_thin_cells():
    tl = _log([(1.0, "2013-01-01", "A"), (1.0, "2014-01-01", "A")])
    ws = windowed_segments(tl, by="grp", window_years=4, start="2012-01-01",
                           end="2015-01-01", min_n=5)
    assert ws.rows["A"][0].n == 2 and ws.rows["A"][0].ok is False
    assert "WINDOWED SEGMENTS" in str(ws)


def test_windows_empty_log_guards():
    tl = TradeLog(pd.DataFrame({"r": [], "entry_date": pd.to_datetime([]), "grp": []}))
    ws = windowed_segments(tl, by="grp")
    assert ws.windows == [] and ws.n_trades == 0


def test_windows_missing_columns_raise():
    tl = _log([(1.0, "2013-01-01", "A")])
    with pytest.raises(ValueError, match="group on"):
        windowed_segments(tl, by="sector")
    no_dates = TradeLog(pd.DataFrame({"r": [1.0], "grp": ["A"]}))
    with pytest.raises(ValueError, match="entry_date"):
        windowed_segments(no_dates, by="grp")
