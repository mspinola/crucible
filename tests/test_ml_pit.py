import pandas as pd
import pytest

from crucible.ml import asof_window, window_before


@pytest.fixture
def daily():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame({"x": range(10)}, index=idx)


def test_asof_includes_cutoff_bar(daily):
    w = asof_window(daily, "2020-01-06", lookback=3)      # idx[5], x==5
    assert list(w["x"]) == [3, 4, 5]                      # on-or-before, last 3


def test_window_before_excludes_cutoff_bar(daily):
    w = window_before(daily, "2020-01-06", lookback=3)    # idx[5]
    assert list(w["x"]) == [2, 3, 4]                      # strictly before, last 3


def test_asof_and_before_differ_by_exactly_the_cutoff_bar(daily):
    a = asof_window(daily, "2020-01-06", lookback=5)
    b = window_before(daily, "2020-01-06", lookback=5)
    assert a["x"].iloc[-1] == 5 and b["x"].iloc[-1] == 4


def test_between_bar_timestamp_reads_through_last_closed_bar(daily):
    w = asof_window(daily, pd.Timestamp("2020-01-06 12:00"), lookback=2)
    assert list(w["x"]) == [4, 5]                          # 01-06 has closed, 01-07 hasn't


def test_lookback_clamps_at_start(daily):
    w = asof_window(daily, "2020-01-03", lookback=100)     # idx[2]
    assert list(w["x"]) == [0, 1, 2]                       # can't go before row 0


def test_nat_cutoff_returns_empty(daily):
    assert window_before(daily, pd.NaT).empty
    assert asof_window(daily, pd.NaT).empty


def test_empty_frame_returns_empty():
    empty = pd.DataFrame({"x": []}, index=pd.DatetimeIndex([]))
    assert asof_window(empty, "2020-01-01").empty


def test_works_on_a_series(daily):
    w = asof_window(daily["x"], "2020-01-06", lookback=3)
    assert isinstance(w, pd.Series)
    assert list(w) == [3, 4, 5]
