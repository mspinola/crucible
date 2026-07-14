import pandas as pd
import pytest

from crucible.edge import TradeLog


def test_from_arrays_minimal():
    tl = TradeLog.from_arrays(r=[1.0, -1.0])
    assert tl.n == 2 and len(tl) == 2
    assert list(tl.r) == [1.0, -1.0]
    assert tl.col("mfe") is None


def test_from_frame_mapping_renames():
    df = pd.DataFrame({"pct_return": [0.5, -0.2], "bars_held": [4, 9]})
    tl = TradeLog.from_frame(df, mapping={"pct_return": "r"})
    assert list(tl.r) == [0.5, -0.2]
    assert tl.col("bars_held") is not None


def test_from_frame_r_col_shorthand():
    df = pd.DataFrame({"R": [1.0, 2.0]})
    tl = TradeLog.from_frame(df, r_col="R")
    assert list(tl.r) == [1.0, 2.0]


def test_missing_r_raises():
    with pytest.raises(ValueError, match="required column"):
        TradeLog(pd.DataFrame({"pnl": [1, 2]}))
