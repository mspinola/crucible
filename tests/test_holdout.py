import numpy as np
import pandas as pd

from crucible.edge import TradeLog
from crucible.validation import holdout, split_train_test


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
