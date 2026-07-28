import numpy as np
import pandas as pd
import pytest

from crucible.edge import detrended_timing_null


def _prices(n=500, drift=0.001, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + rets))


def test_returns_null_of_requested_length():
    null = detrended_timing_null(_prices(), holds=[5] * 30, n_samples=300, seed=1)
    assert null.shape == (300,)
    assert np.all(np.isfinite(null))


def test_detrend_centers_the_null_near_zero():
    # with a strong up-drift, the *detrended* null should sit near zero, while the
    # un-detrended null inherits the drift and sits clearly positive.
    px = _prices(n=800, drift=0.002)
    holds = [10] * 40
    centered = detrended_timing_null(px, holds, n_samples=800, detrend=True, seed=2).mean()
    drifted = detrended_timing_null(px, holds, n_samples=800, detrend=False, seed=2).mean()
    assert abs(centered) < abs(drifted)
    assert drifted > 0


def test_direction_flips_sign_of_the_null_mean():
    px = _prices(n=600, drift=0.003)
    holds = [8] * 50
    longs = detrended_timing_null(px, holds, directions=[1] * 50, detrend=False, seed=3).mean()
    shorts = detrended_timing_null(px, holds, directions=[-1] * 50, detrend=False, seed=3).mean()
    assert longs > 0 > shorts


def test_reproducible_with_seed():
    px = _prices()
    a = detrended_timing_null(px, [5] * 20, n_samples=200, seed=7)
    b = detrended_timing_null(px, [5] * 20, n_samples=200, seed=7)
    assert np.array_equal(a, b)


def test_too_few_bars_returns_empty():
    assert detrended_timing_null(pd.Series([100.0, 101.0]), holds=[1, 1]).size == 0


def test_accepts_array_and_handles_nan_holds():
    px = _prices().to_numpy()
    null = detrended_timing_null(px, holds=[np.nan, 5, 3], n_samples=100, seed=0)
    assert null.shape == (100,) and np.all(np.isfinite(null))


def test_scale_none_is_unchanged():
    # scale=None (a simple-return log) must be identical to the pre-scale behavior.
    px = _prices(n=600)
    holds = [7] * 30
    a = detrended_timing_null(px, holds, n_samples=250, seed=9)
    b = detrended_timing_null(px, holds, scale=None, n_samples=250, seed=9)
    assert np.array_equal(a, b)


def test_scale_denominates_the_null_linearly():
    # a constant per-trade scale multiplies every draw by that constant, so the whole
    # null scales linearly — this is the exact fractional->R conversion when the scale
    # is entry/risk (R = fractional_return * entry/risk).
    px = _prices(n=600, drift=0.001)
    holds = [8] * 40
    base = detrended_timing_null(px, holds, n_samples=300, seed=5)
    scaled = detrended_timing_null(px, holds, scale=[3.0] * 40, n_samples=300, seed=5)
    assert np.allclose(scaled, 3.0 * base)


def test_scale_length_must_match_trades():
    with pytest.raises(ValueError):
        detrended_timing_null(_prices(), holds=[5] * 10, scale=[1.0] * 9)
