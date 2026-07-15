import numpy as np

from crucible.validation import fold_dispersion, walk_forward_efficiency


def test_fold_dispersion_consistent_folds_pass():
    d = fold_dispersion([1.8, 2.1, 1.6, 2.0])
    assert d["n_folds"] == 4
    assert d["pct_folds_tradable"] == 1.0
    assert d["passes"] is True


def test_fold_dispersion_rejects_when_half_folds_untradable():
    d = fold_dispersion([2.0, 1.8, -0.5, -1.2], min_tradable_pct=0.5)
    assert d["pct_folds_tradable"] == 0.5      # exactly at the bar
    # cv is what decides here; two positive + two negative -> high dispersion
    assert d["fold_sqn_cv"] > d["max_cv"]
    assert d["passes"] is False


def test_fold_dispersion_zero_mean_gives_inf_cv_and_fails():
    d = fold_dispersion([2.0, -2.0])
    assert d["fold_sqn_cv"] == float("inf")
    assert d["passes"] is False


def test_fold_dispersion_drops_none_and_nan():
    d = fold_dispersion([1.5, None, np.nan, 1.7])
    assert d["n_folds"] == 2


def test_fold_dispersion_empty_is_none():
    assert fold_dispersion([]) is None
    assert fold_dispersion([None, np.nan]) is None


def test_wfe_in_band_passes_and_hits_target():
    w = walk_forward_efficiency([0.6, 0.7, 0.65])
    assert w["passes"] is True
    assert w["in_target_band"] is True
    assert w["aggregate_wfe"] == np.mean([0.6, 0.7, 0.65])


def test_wfe_too_low_rejected():
    w = walk_forward_efficiency([0.1, 0.2, 0.15])
    assert w["passes"] is False
    assert w["in_target_band"] is False


def test_wfe_too_high_is_also_rejected():
    # above reject_high (1.00) is "too good to be true", not a pass
    w = walk_forward_efficiency([1.5, 2.0, 1.2])
    assert w["aggregate_wfe"] > w["reject_high"]
    assert w["passes"] is False


def test_wfe_empty_is_none():
    assert walk_forward_efficiency([]) is None
