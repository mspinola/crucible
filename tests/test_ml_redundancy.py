import numpy as np
import pandas as pd
import pytest

from crucible.ml import RedundancyReport, cramers_v, fold_ic, redundancy_droplist

# --- cramers_v -----------------------------------------------------------------

def test_cramers_v_identical_is_one():
    a = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    assert cramers_v(a, a.copy()) == pytest.approx(1.0)


def test_cramers_v_independent_near_zero():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.integers(0, 2, 2000))
    b = pd.Series(rng.integers(0, 2, 2000))
    assert cramers_v(a, b) < 0.1


def test_cramers_v_constant_series_is_zero():
    a = pd.Series([1, 1, 1, 1])
    b = pd.Series([0, 1, 0, 1])
    assert cramers_v(a, b) == 0.0


# --- fold_ic -------------------------------------------------------------------

def _ic_panel(n=120, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    return pd.DataFrame({
        "good": base,                              # == target, IC ~ 1
        "noise": rng.normal(size=n),               # unrelated, IC ~ 0
        "target": base,
    })


def test_fold_ic_ranks_signal_over_noise():
    ic = fold_ic(_ic_panel(), ["good", "noise"], target="target", n_splits=5)
    row = ic.set_index("feature")
    assert row.loc["good", "ic_mean"] == pytest.approx(1.0, abs=1e-9)
    assert abs(row.loc["noise", "ic_mean"]) < 0.3
    assert row.loc["good", "n_folds"] == 5          # 120 rows / 5 folds = 24 >= min_test
    assert ic.iloc[0]["feature"] == "good"          # sorted by ic_abs desc


def test_fold_ic_respects_groups():
    p = _ic_panel(n=200)
    p["sym"] = (["A"] * 100) + (["B"] * 100)
    ic = fold_ic(p, ["good", "noise"], target="target", group="sym", n_splits=5)
    # 2 groups x 5 folds, each fold 20 rows >= min_test
    assert ic.set_index("feature").loc["good", "n_folds"] == 10


def test_fold_ic_skips_short_groups():
    p = _ic_panel(n=30)                             # < min_group (40)
    ic = fold_ic(p, ["good"], target="target")
    assert ic.set_index("feature").loc["good", "n_folds"] == 0
    assert np.isnan(ic.set_index("feature").loc["good", "ic_mean"])


# --- redundancy_droplist -------------------------------------------------------

def test_droplist_keeps_highest_ic_of_a_continuous_cluster():
    rng = np.random.default_rng(2)
    base = rng.normal(size=150)
    panel = pd.DataFrame({
        "f_exact": base,                            # == target -> highest IC, should KEEP
        "f_noisy1": base + rng.normal(0, 0.05, 150),
        "f_noisy2": base + rng.normal(0, 0.05, 150),
        "f_indep": rng.normal(size=150),            # uncorrelated -> not in any group
        "target": base,
    })
    rep = redundancy_droplist(panel, ["f_exact", "f_noisy1", "f_noisy2", "f_indep"],
                              target="target", corr_thresh=0.85)
    assert isinstance(rep, RedundancyReport)
    assert set(rep.kept) == {"f_exact"}
    assert set(rep.dropped) == {"f_noisy1", "f_noisy2"}
    assert "f_indep" not in rep.dropped and "f_indep" not in rep.kept


def test_droplist_groups_redundant_binaries():
    rng = np.random.default_rng(3)
    base = rng.normal(size=150)
    b = (base > 0).astype(int)
    panel = pd.DataFrame({
        "b_keep": b,                                # mirrors target sign -> higher IC
        "b_mirror": b,                              # identical -> Cramér's V = 1
        "target": base,
    })
    rep = redundancy_droplist(panel, ["b_keep", "b_mirror"], target="target",
                              v_thresh=0.8, binary_max_card=3)
    assert len(rep.dropped) == 1 and len(rep.kept) == 1
    assert set(rep.kept + rep.dropped) == {"b_keep", "b_mirror"}


def test_droplist_empty_when_nothing_redundant():
    rng = np.random.default_rng(4)
    panel = pd.DataFrame({
        "a": rng.normal(size=120),
        "b": rng.normal(size=120),
        "target": rng.normal(size=120),
    })
    rep = redundancy_droplist(panel, ["a", "b"], target="target", corr_thresh=0.85)
    assert rep.droplist.empty
    assert rep.dropped == [] and rep.kept == []
