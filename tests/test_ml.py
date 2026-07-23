import numpy as np
import pandas as pd
import pytest

from crucible.ml import (
    AlphaGateError,
    alpha_gate,
    information_coefficient,
    quantile_decay,
)

# --- information_coefficient ---------------------------------------------------

def test_ic_perfect_positive():
    preds = pd.DataFrame({"score": [1, 2, 3, 4, 5, 6], "label": [1, 2, 3, 4, 5, 6]})
    assert information_coefficient(preds) == pytest.approx(1.0)


def test_ic_perfect_negative():
    preds = pd.DataFrame({"score": [1, 2, 3, 4, 5, 6], "label": [6, 5, 4, 3, 2, 1]})
    assert information_coefficient(preds) == pytest.approx(-1.0)


def test_ic_invariant_to_label_encoding():
    rng = np.random.default_rng(0)
    score = rng.normal(size=200)
    signed = np.where(score + rng.normal(0, 0.5, 200) > 0, 1, -1)   # +1/-1
    binary = (signed > 0).astype(int)                               # 0/1 of the same wins
    ic_signed = information_coefficient(pd.DataFrame({"score": score, "label": signed}))
    ic_binary = information_coefficient(pd.DataFrame({"score": score, "label": binary}))
    assert ic_signed == pytest.approx(ic_binary)                    # rank-based -> identical


def test_ic_missing_column_is_zero():
    assert information_coefficient(pd.DataFrame({"score": [1, 2, 3]})) == 0.0


def test_ic_too_few_rows_is_zero():
    preds = pd.DataFrame({"score": [1, 2, 3, 4], "label": [1, 2, 3, 4]})
    assert information_coefficient(preds) == 0.0


def test_ic_custom_column_names():
    preds = pd.DataFrame({"prob": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
    assert information_coefficient(preds, score="prob", label="y") == pytest.approx(1.0)


# --- alpha_gate ----------------------------------------------------------------

def test_alpha_gate_passes_at_or_above():
    alpha_gate(0.05, min_ic=0.05)      # equal -> no raise
    alpha_gate(0.20, min_ic=0.05)


def test_alpha_gate_raises_below():
    with pytest.raises(AlphaGateError):
        alpha_gate(0.01, min_ic=0.05)


# --- quantile_decay ------------------------------------------------------------

def _decay_frame(losers=-1):
    # score 0..99; wins (label>0) concentrated in the top scores so win rate
    # rises across quintiles: [0.0, 0.0, 0.5, 1.0, 1.0]
    score = np.arange(100)
    label = np.where(score >= 50, 1, losers)
    return pd.DataFrame({"score": score, "label": label})


def test_quantile_decay_monotonic_and_spread():
    d = quantile_decay(_decay_frame(), q=5)
    assert d.monotonic is True
    assert list(d.table["win_rate"]) == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert d.spread == pytest.approx(1.0)
    assert len(d.table) == 5
    assert list(d.table["count"]) == [20, 20, 20, 20, 20]


def test_quantile_decay_win_defined_by_sign_not_encoding():
    # losers as -1 vs 0 must give identical win rates (the bug the original had)
    a = quantile_decay(_decay_frame(losers=-1))
    b = quantile_decay(_decay_frame(losers=0))
    assert list(a.table["win_rate"]) == list(b.table["win_rate"])


def test_quantile_decay_flat_signal_not_monotonic_zero_spread():
    rng = np.random.default_rng(3)
    preds = pd.DataFrame({"score": rng.normal(size=500), "label": rng.choice([-1, 1], 500)})
    d = quantile_decay(preds)
    assert d.spread == pytest.approx(0.0, abs=0.15)   # no ordering -> ~flat


def test_quantile_decay_missing_column_raises():
    with pytest.raises(ValueError):
        quantile_decay(pd.DataFrame({"score": [1, 2, 3, 4, 5]}))


def test_quantile_decay_too_few_rows_raises():
    preds = pd.DataFrame({"score": [1, 2, 3], "label": [1, -1, 1]})
    with pytest.raises(ValueError):
        quantile_decay(preds, q=5)
