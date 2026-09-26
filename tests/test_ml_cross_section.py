"""rank_band_decay and random_selection_null on synthetic cross-sections.

The first half checks the tools find a planted signal. The second half pins the three
ways the first draft passed a worthless ranking: bands averaged over different years,
a null that redrew names independently each date (sticky score, overlapping labels),
and a delisted name silently dropped.
"""
import numpy as np
import pandas as pd
import pytest

from crucible.edge import block_bootstrap_pvalue
from crucible.ml import random_selection_null, rank_band_decay


def _panel(signal: float, *, dates=120, names=40, seed=0, cliff=False):
    """Monthly panel: shared market move + idiosyncratic noise + optional signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(dates):
        mkt = rng.normal(0.008, 0.04)
        score = rng.normal(size=names)
        if cliff:   # edge only in the top 5 names by score
            boost = np.where(score >= np.sort(score)[-5], signal, 0.0)
        else:
            boost = signal * score
        label = mkt + boost + rng.normal(0, 0.08, names)
        rows += [{"date": t, "name": i, "score": score[i], "label": label[i]}
                 for i in range(names)]
    return pd.DataFrame(rows)


def _wide_to_panel(score: np.ndarray, label: np.ndarray) -> pd.DataFrame:
    t, n = score.shape
    return pd.DataFrame({"date": np.repeat(np.arange(t), n), "name": np.tile(np.arange(n), t),
                         "score": score.ravel(), "label": label.ravel()})


def _sticky_drift_panel(seed, dates=120, names=40):
    """A near-constant score, and per-name drift the score knows nothing about."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0, 0.01, names)
    base = rng.normal(size=names)
    score = base + 0.05 * rng.normal(size=(dates, names))
    label = drift + rng.normal(0, 0.04, (dates, names))
    return _wide_to_panel(score, label)


def _overlap_panel(seed, dates=240, names=30, hold=20):
    """Dates spaced one bar apart, labels = the next `hold` bars: consecutive labels share
    19 of 20 bars. The score is a random walk, sticky and uninformative."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, (dates + hold, names))
    fwd = np.stack([ret[t:t + hold].sum(0) for t in range(dates)])
    score = np.cumsum(rng.normal(size=(dates, names)), axis=0)
    return _wide_to_panel(score, fwd)


# ---------------------------------------------------------------- finds a real signal

def test_decay_detects_planted_slope():
    d = rank_band_decay(_panel(0.02), band_size=5, n_bands=6)
    assert list(d.table["ranks"]) == ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"]
    assert d.spread > 0.03
    assert d.table["mean_return"].iloc[0] == d.table["mean_return"].max()


def test_cliff_share_separates_cliff_from_slope():
    slope = rank_band_decay(_panel(0.02, seed=1), band_size=5)
    cliff = rank_band_decay(_panel(0.05, seed=1, cliff=True), band_size=5)
    assert cliff.cliff_share > 0.7
    assert slope.cliff_share < cliff.cliff_share


def test_partial_bands_are_dropped():
    panel = _panel(0.0, dates=3, names=12)
    d = rank_band_decay(panel, band_size=5, n_bands=None)
    assert list(d.table["band"]) == [1, 2]          # ranks 11-12 never fill a band
    assert (d.table["names"] == 15).all()
    assert d.dates_dropped == 0


def test_null_rejects_real_ranking():
    res = random_selection_null(_panel(0.02, seed=2), top_n=5, n_sims=1000)
    assert res.p_value < 0.01
    assert res.observed > res.null_mean
    assert block_bootstrap_pvalue(res.excess.to_numpy(), block=6) < 0.05


def test_null_does_not_reject_noise_even_in_rising_market():
    res = random_selection_null(_panel(0.0, seed=3), top_n=5, n_sims=1000)
    assert res.observed > 0          # the universe went up...
    assert res.p_value > 0.05        # ...but the ordering added nothing


def test_null_is_deterministic_and_validates():
    p = _panel(0.01, dates=24, seed=4)
    a = random_selection_null(p, top_n=5, n_sims=200, seed=7)
    b = random_selection_null(p, top_n=5, n_sims=200, seed=7)
    assert a.p_value == b.p_value
    with pytest.raises(ValueError):
        random_selection_null(p, top_n=500)
    with pytest.raises(ValueError):
        random_selection_null(p.drop(columns="name"), top_n=5)
    with pytest.raises(ValueError):
        random_selection_null(pd.concat([p, p.iloc[:1]]), top_n=5)   # duplicate row
    with pytest.raises(ValueError):
        rank_band_decay(p.drop(columns="label"))


# ------------------------------------------------------- refuses a worthless ranking

def test_bands_are_compared_on_the_same_dates():
    """Noise score, universe growing 10 -> 40 names, a rally only in the early dates.
    Averaging each band over the dates it happens to be full on handed the top bands
    the rally: a 1.7% spread at t = 7 on a meaningless score."""
    rng = np.random.default_rng(0)
    rows = []
    for t in range(120):
        n = 10 + (30 * t) // 119
        mkt = 0.05 if t < 40 else 0.0
        s, lab = rng.normal(size=n), mkt + rng.normal(0, 0.02, n)
        rows += [{"date": t, "score": s[i], "label": lab[i]} for i in range(n)]
    d = rank_band_decay(pd.DataFrame(rows), band_size=5, n_bands=6)
    assert d.table["periods"].nunique() == 1
    assert d.dates_dropped == 120 - d.table["periods"].iloc[0]
    assert abs(d.spread) < 0.005
    assert d.table["t_stat"].abs().max() < 3


@pytest.mark.parametrize("make", [_sticky_drift_panel, _overlap_panel],
                         ids=["sticky-score-name-drift", "overlapping-labels"])
def test_null_holds_its_size_on_dependent_panels(make):
    """A null that redrew names independently each date rejected these worthless
    rankings 29% and 43% of the time at the 5% level. The name permutation keeps
    the dependence, so the false-positive rate stays near nominal."""
    ps = np.array([random_selection_null(make(seed), top_n=5, n_sims=200, seed=seed).p_value
                   for seed in range(40)])
    assert (ps < 0.05).mean() <= 0.125      # 5 of 40; binomial(40, .05) exceeds it ~5%


def test_missing_label_is_refused_not_dropped():
    p = _panel(0.0, dates=12, seed=5)
    p.loc[(p["date"] == 3) & (p["name"] == 7), "label"] = np.nan   # delisted mid-period
    with pytest.raises(ValueError, match="delisting"):
        rank_band_decay(p)
    with pytest.raises(ValueError, match="delisting"):
        random_selection_null(p, top_n=5, n_sims=50)
    assert rank_band_decay(p, missing_label="drop").labels_dropped == 1
    assert random_selection_null(p, top_n=5, n_sims=50, missing_label="drop").labels_dropped == 1


def test_unrealized_final_dates_are_not_missing_labels():
    """The last holding period has not closed yet: every label on the date is empty."""
    p = _panel(0.0, dates=12, seed=6)
    p.loc[p["date"] == 11, "label"] = np.nan
    assert random_selection_null(p, top_n=5, n_sims=50).periods == 11
    assert rank_band_decay(p).table["periods"].iloc[0] == 11


def test_unbalanced_panel_scores_every_simulation_on_the_same_dates():
    p = _panel(0.02, dates=36, names=20, seed=8)
    p = p[~((p["name"] < 6) & (p["date"] < 12))]          # six names list late
    res = random_selection_null(p, top_n=5, n_sims=300)
    assert res.periods == 36
    assert res.p_value < 0.05
