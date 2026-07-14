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
