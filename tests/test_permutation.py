import numpy as np
import pytest

from crucible.edge import TradeLog
from crucible.validation import (
    sign_permutation_pvalue, sidak_correction, whites_reality_check,
)


def test_sign_permutation_strong_edge_low_p():
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    p = sign_permutation_pvalue(tl, n_permutations=2000, seed=0)
    assert p < 0.05


def test_sign_permutation_no_edge_high_p():
    p = sign_permutation_pvalue([1.0, -1.0] * 50, n_permutations=2000, seed=0)
    assert p > 0.2


def test_sidak_monotonic():
    assert sidak_correction(0.01, 1) == pytest.approx(0.01)
    assert sidak_correction(0.01, 50) > sidak_correction(0.01, 5)
    assert sidak_correction(0.01, 100) <= 1.0


def test_whites_reality_check_picks_best_and_corrects():
    rng = np.random.default_rng(1)
    variants = {f"noise_{i}": rng.normal(0, 1, 120) for i in range(9)}
    variants["real"] = rng.normal(0.6, 1.0, 120)     # the one true edge
    out = whites_reality_check(variants, n_permutations=1500, seed=0)
    assert out["n_variants"] == 10
    assert out["best_variant"] == "real"
    assert 0.0 < out["corrected_pvalue"] <= 1.0


# --- SPA (Hansen) — WRC's more powerful successor ---------------------------

from crucible.validation import spa_test


def test_spa_finds_real_edge_low_p():
    rng = np.random.default_rng(3)
    variants = {f"noise_{i}": rng.normal(0, 1, 150) for i in range(8)}
    variants["real"] = rng.normal(0.6, 1.0, 150)
    res = spa_test(variants, n_permutations=3000, seed=0)
    assert res["best_variant"] == "real"
    assert res["corrected_pvalue"] < 0.05


def test_spa_more_powerful_than_wrc_with_junk():
    # One real edge padded with several clearly-inferior, high-variance junk
    # variants. WRC lets the junk inflate the null max; SPA studentizes and
    # EXCLUDES it -> SPA p <= WRC p (and it reports the exclusions).
    rng = np.random.default_rng(4)
    variants = {"real": rng.normal(0.5, 1.0, 150)}
    for i in range(6):
        variants[f"junk_{i}"] = rng.normal(-2.0, 3.0, 150)   # strongly negative, volatile
    wrc = whites_reality_check(variants, n_permutations=3000, seed=0)
    spa = spa_test(variants, n_permutations=3000, seed=0)
    assert spa["n_excluded"] >= 1                     # junk dropped as inferior
    assert spa["corrected_pvalue"] <= wrc["corrected_pvalue"] + 1e-9


def test_spa_empty_guard():
    res = spa_test({}, n_permutations=100)
    assert res["best_variant"] is None and res["corrected_pvalue"] == 1.0
