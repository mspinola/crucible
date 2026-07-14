import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlc():
    """A synthetic daily OHLC frame: gently up-drifting random walk, ~6 years."""
    rng = np.random.default_rng(7)
    n = 1500
    rets = rng.normal(0.0004, 0.01, n)          # small positive drift
    close = 100 * np.cumprod(1 + rets)
    open_ = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, 0.006, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)
