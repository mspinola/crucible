import numpy as np
import pandas as pd
import pytest

from crucible.edge import TradeLog, detrended_timing_null, expectancy
from crucible.validation import Thresholds, gate_real


def _ohlc(n=800, drift=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(drift, 0.01, n)),
                      index=pd.date_range("2015-01-01", periods=n, freq="B"))
    span = np.abs(rng.normal(0, 0.004, n)) * close.values
    return pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]),
                         "High": close + span, "Low": close - span, "Close": close},
                        index=close.index)


def _check(gate, name):
    return next(c for c in gate.checks if c.name == name)


def test_detrended_null_matches_the_primitive():
    # gate_real's detrended threshold must equal the 95th pct of the primitive,
    # called with the same holds/directions/seed — i.e. it truly delegates.
    rng = np.random.default_rng(1)
    r = rng.normal(0.01, 0.02, 120)
    holds = rng.integers(5, 25, 120)
    dirs = rng.choice([1.0, -1.0], 120)
    tl = TradeLog.from_arrays(r=r, bars_held=holds)
    ohlc = _ohlc()
    thr = Thresholds(n_random_sims=300, seed=7)

    g = gate_real(tl, prices=ohlc, null="detrended", directions=dirs,
                  n_variants=1, thr=thr)
    chk = _check(g, "beats_random_timing")

    expected = detrended_timing_null(ohlc["Close"], holds, directions=dirs,
                                     n_samples=300, seed=7)
    assert chk.threshold == pytest.approx(float(np.percentile(expected, 95)))
    assert chk.value == pytest.approx(expectancy(r))
    assert "detrended" in chk.detail


def test_detrended_and_random_entry_are_different_nulls():
    rng = np.random.default_rng(2)
    tl = TradeLog.from_arrays(r=rng.normal(0.008, 0.02, 100),
                              bars_held=rng.integers(5, 20, 100))
    ohlc = _ohlc(drift=0.001)                     # a drift-bearing asset
    thr = Thresholds(n_random_sims=300, seed=0)
    d = _check(gate_real(tl, prices=ohlc, null="detrended", n_variants=1, thr=thr),
               "beats_random_timing").threshold
    re = _check(gate_real(tl, prices=ohlc, null="random_entry", n_variants=1, thr=thr),
                "beats_random_timing").threshold
    assert d != re                                 # different constructions -> different bars


def test_default_null_is_random_entry_unchanged():
    rng = np.random.default_rng(3)
    tl = TradeLog.from_arrays(r=rng.normal(0.01, 0.02, 80), bars_held=[10] * 80)
    ohlc = _ohlc()
    thr = Thresholds(n_random_sims=200, seed=0)
    explicit = _check(gate_real(tl, prices=ohlc, null="random_entry", n_variants=1, thr=thr),
                      "beats_random_timing").threshold
    default = _check(gate_real(tl, prices=ohlc, n_variants=1, thr=thr),
                     "beats_random_timing").threshold
    assert default == explicit                     # default behavior preserved
