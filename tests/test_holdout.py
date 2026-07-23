import numpy as np
import pandas as pd
import pytest

from crucible.edge import TradeLog
from crucible.edge.stats import reality_check
from crucible.validation import (
    full_sample,
    holdout,
    segmented_holdout,
    split_train_test,
)


def _dated_log(r, entries, exits):
    return TradeLog.from_arrays(
        r=r,
        entry_date=pd.to_datetime(entries).values,
        exit_date=pd.to_datetime(exits).values,
    )


def test_split_is_leakage_controlled():
    # trade 1 fully before split; trade 2 straddles it (enters before, exits after)
    # -> must be excluded from train; trade 3 is in the embargo; trade 4 is test.
    tl = _dated_log(
        r=[1.0, 1.0, 1.0, 1.0],
        entries=["2018-01-01", "2018-12-20", "2019-01-10", "2019-06-01"],
        exits=["2018-02-01", "2019-01-15", "2019-01-20", "2019-06-20"],
    )
    train, test = split_train_test(tl, "2019-01-01", embargo_weeks=8)
    assert train.n == 1                     # only the fully-early trade
    assert test.n == 1                      # only the post-embargo trade


def test_holdout_verdicts_run():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=400, freq="7D")
    r = rng.normal(0.3, 1.0, 400)           # positive-ish edge
    tl = TradeLog.from_arrays(r=r, entry_date=dates.values,
                              exit_date=(dates + pd.Timedelta(days=1)).values)
    res = holdout(tl, "2019-01-01", embargo_weeks=8, n_boot=1000)
    assert res.train_n > 0 and res.test_n > 0
    assert res.label == res.test.label       # verdict is the OOS side
    assert res.label in ("HELD", "FRAGILE", "FAIL")
    assert "HOLDOUT" in str(res)


def _segmented_log(seed=0, n=600):
    """A dated log tagged with an 'asset_class' column spanning early/late."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="5D")
    classes = np.array(["Metals", "Energy", "Equities"])[rng.integers(0, 3, n)]
    df = pd.DataFrame({
        "r": rng.normal(0.2, 1.0, n),
        "entry_date": dates,
        "exit_date": dates + pd.Timedelta(days=1),
        "asset_class": classes,
    })
    return TradeLog(df)


def test_segmented_holdout_slices_match_direct_holdout():
    tl = _segmented_log()
    seg = segmented_holdout(tl, by="asset_class", split="2019-01-01",
                            embargo_weeks=8, n_boot=800, seed=1)
    # every distinct class became a segment
    assert set(seg.segments) == set(tl.frame["asset_class"].unique())
    # a segment is EXACTLY the same holdout run on the pre-filtered log
    sub = TradeLog(tl.frame[tl.frame["asset_class"] == "Metals"].reset_index(drop=True))
    direct = holdout(sub, "2019-01-01", embargo_weeks=8, n_boot=800, seed=1)
    metals = seg.segments["Metals"]
    assert metals.test_n == direct.test_n and metals.train_n == direct.train_n
    assert metals.test.point == pytest.approx(direct.test.point)
    assert metals.test.ci.low == pytest.approx(direct.test.ci.low)


def test_segmented_holdout_segment_counts_reconcile_with_overall():
    tl = _segmented_log()
    seg = segmented_holdout(tl, by="asset_class", split="2019-01-01", n_boot=200)
    # no trade is lost or double-counted across the partition
    assert sum(s.test_n for s in seg.segments.values()) == seg.overall.test_n
    assert sum(s.train_n for s in seg.segments.values()) == seg.overall.train_n


def test_segmented_holdout_leakage_control_per_segment():
    # one Metals trade straddles the split; it must be dropped from that
    # segment's TRAIN, just as split_train_test does for the whole book.
    df = pd.DataFrame({
        "r": [1.0, 1.0, 1.0],
        "entry_date": pd.to_datetime(["2018-01-01", "2018-12-20", "2019-06-01"]),
        "exit_date": pd.to_datetime(["2018-02-01", "2019-01-15", "2019-06-20"]),
        "asset_class": ["Metals", "Metals", "Metals"],
    })
    seg = segmented_holdout(TradeLog(df), by="asset_class", split="2019-01-01",
                            embargo_weeks=8, n_boot=100)
    assert seg.segments["Metals"].train_n == 1   # straddler excluded
    assert seg.segments["Metals"].test_n == 1


def test_segmented_holdout_flags_thin_segments():
    tl = _segmented_log()
    seg = segmented_holdout(tl, by="asset_class", split="2019-01-01",
                            n_boot=100, min_n=10_000)   # nothing clears it
    assert seg.reliable() == {}
    assert set(seg.thin()) == set(seg.segments)
    assert "SEGMENTED HOLDOUT" in str(seg)


def test_segmented_holdout_missing_column_raises():
    tl = _segmented_log()
    with pytest.raises(ValueError, match="group on"):
        segmented_holdout(tl, by="sector", split="2019-01-01")


def test_full_sample_is_whole_book_reality_check():
    tl = _segmented_log()
    v = full_sample(tl, n_boot=500, seed=3)
    direct = reality_check(tl, n_boot=500, seed=3)
    # names the intent; numerically identical to a whole-book reality_check
    assert v.point == pytest.approx(direct.point)
    assert v.ci.low == pytest.approx(direct.ci.low)
    assert v.label == direct.label
