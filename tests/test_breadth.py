import numpy as np
import pandas as pd
import pytest

from crucible.breadth import Breadth, effective_n, participation_ratio


def _panel(cols: dict, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A dates x assets return panel from named 1-D arrays of length ``n``."""
    idx = pd.date_range("2015-01-01", periods=n, freq="W")
    return pd.DataFrame(cols, index=idx)


# --- participation_ratio: the exact known-answer anchors -----------------------

def test_participation_ratio_independent_equals_n():
    # N equal eigenvalues (identity correlation) -> N_eff == N
    assert participation_ratio(np.ones(5)) == pytest.approx(5.0)


def test_participation_ratio_single_factor_equals_one():
    # all variance in one eigenvalue (perfect correlation) -> N_eff == 1
    assert participation_ratio(np.array([5.0, 0.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_participation_ratio_zero_raises():
    with pytest.raises(ValueError):
        participation_ratio(np.zeros(3))


# --- effective_n on real return panels -----------------------------------------

def test_effective_n_independent_streams_near_n():
    rng = np.random.default_rng(1)
    cols = {f"m{i}": rng.normal(0, 0.01, 400) for i in range(6)}
    b = effective_n(_panel(cols))
    assert b.n_assets == 6
    assert b.n_eff == pytest.approx(6.0, abs=0.6)   # ~independent -> N_eff ~ N
    assert b.redundancy == pytest.approx(6 / b.n_eff)


def test_effective_n_perfectly_correlated_is_one():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.01, 400)
    cols = {f"m{i}": base.copy() for i in range(4)}   # identical streams
    b = effective_n(_panel(cols))
    assert b.n_eff == pytest.approx(1.0, abs=1e-6)


def test_effective_n_two_blocs_between_bounds():
    # two internally-correlated blocs of 3 -> ~2 independent bets, well below 6
    rng = np.random.default_rng(3)
    dollar = rng.normal(0, 0.01, 400)
    grains = rng.normal(0, 0.01, 400)
    cols = {}
    for i in range(3):
        cols[f"fx{i}"] = dollar + rng.normal(0, 0.001, 400)
        cols[f"ag{i}"] = grains + rng.normal(0, 0.001, 400)
    b = effective_n(_panel(cols))
    assert 1.5 < b.n_eff < 3.5
    assert b.n_eff < b.n_assets


def test_loadings_shape_and_alignment():
    rng = np.random.default_rng(4)
    cols = {f"m{i}": rng.normal(0, 0.01, 400) for i in range(4)}
    b = effective_n(_panel(cols))
    assert isinstance(b, Breadth)
    assert b.loadings.shape == (4, 4)
    assert list(b.loadings.index) == list(cols)
    assert list(b.loadings.columns) == ["PC1", "PC2", "PC3", "PC4"]
    # eigenvalues descending and summing to N (trace of a correlation matrix)
    assert np.all(np.diff(b.eigenvalues) <= 1e-9)
    assert b.eigenvalues.sum() == pytest.approx(4.0)


def test_short_history_streams_dropped():
    rng = np.random.default_rng(5)
    cols = {f"m{i}": rng.normal(0, 0.01, 400) for i in range(3)}
    short = np.full(400, np.nan)
    short[-10:] = rng.normal(0, 0.01, 10)             # only 10 obs
    cols["late"] = short
    b = effective_n(_panel(cols), min_obs=30)
    assert "late" not in b.corr.columns
    assert b.n_assets == 3


def test_too_few_streams_raises():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        effective_n(_panel({"only": rng.normal(0, 0.01, 400)}))
