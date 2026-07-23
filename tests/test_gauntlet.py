from types import SimpleNamespace

import numpy as np

from crucible.edge import TradeLog, barrier_trades
from crucible.strategies import ma_cross
from crucible.validation import (
    Thresholds,
    gate_durable,
    gate_general,
    gate_real,
    gate_strong,
    run_gauntlet,
)

# small resampling budgets so the suite stays fast
FAST = Thresholds(n_boot=1500, n_perm=1500, n_random_sims=150)


# ── STRONG ────────────────────────────────────────────────────────────────────
def test_strong_passes_on_a_clear_edge():
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    g = gate_strong(tl, thr=FAST)
    assert g.passed is True
    names = {c.name for c in g.checks}
    assert {"expectancy_ci_lower", "profit_factor_ci_lower"} <= names


def test_strong_fails_when_expectancy_ci_straddles_zero():
    tl = TradeLog.from_arrays(r=[1.0, -1.0] * 40)     # expectancy ~0, CI includes 0
    g = gate_strong(tl, thr=FAST)
    assert g.passed is False
    assert "expectancy_ci_lower" in {c.name for c in g.failed_checks}


def test_strong_sqn_is_soft():
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    g = gate_strong(tl, thr=FAST)
    sqn_check = next(c for c in g.checks if c.name == "sqn_ci_lower")
    assert sqn_check.hard is False


# ── REAL ──────────────────────────────────────────────────────────────────────
def test_real_passes_on_a_clear_edge_no_prices():
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    g = gate_real(tl, thr=FAST)
    assert g.passed is True                          # permutation p is the hard check
    # no prices -> beats_random_timing recorded but soft, so it can't gate
    brt = next(c for c in g.checks if c.name == "beats_random_timing")
    assert brt.hard is False


def test_real_fails_on_no_edge():
    tl = TradeLog.from_arrays(r=[1.0, -1.0] * 60)
    assert gate_real(tl, thr=FAST).passed is False


def test_real_sidak_correction_named_and_harder():
    tl = TradeLog.from_arrays(r=[2.0, 2.0, -1.0, -1.0, 2.0, -1.0] * 12)
    raw = gate_real(tl, thr=FAST)
    corr = gate_real(tl, n_variants=200, thr=FAST)
    p_raw = next(c for c in raw.checks if c.name == "permutation_pvalue").value
    p_corr = next(c for c in corr.checks if c.name == "corrected_pvalue").value
    assert p_corr >= p_raw                           # correcting for the search only hurts


def test_real_uses_reality_check_when_variant_returns_given():
    rng = np.random.default_rng(1)
    variants = {f"noise_{i}": rng.normal(0, 1, 120) for i in range(8)}
    variants["real"] = rng.normal(0.7, 1.0, 120)
    tl = TradeLog.from_arrays(r=variants["real"])
    g = gate_real(tl, variant_returns=variants, thr=FAST)
    assert "reality_check_pvalue" in {c.name for c in g.checks}


def test_real_beats_random_timing_is_hard_when_prices_given(ohlc):
    entries = ma_cross(ohlc, fast=10, slow=30)
    tl = barrier_trades(ohlc, entries, side="long", tp=2.0, sl=1.0, timeout=20)
    g = gate_real(tl, prices=ohlc, side="long", tp=2.0, sl=1.0, thr=FAST)
    brt = next(c for c in g.checks if c.name == "beats_random_timing")
    assert brt.hard is True                          # a price series makes it gate
    assert brt.value is not None


# ── DURABLE ───────────────────────────────────────────────────────────────────
def _fake_wf(folds):
    """A stand-in WalkForwardResult: only .folds (each with .wfe and .oos_trades)."""
    return SimpleNamespace(folds=[
        SimpleNamespace(wfe=wfe, oos_trades=TradeLog.from_arrays(r=r))
        for wfe, r in folds
    ])


def test_durable_passes_on_healthy_folds():
    wf = _fake_wf([(0.6, [3, 3, -1, -1] * 6), (0.7, [3, 3, 3, -1] * 6),
                   (0.55, [2, 2, -1, -1] * 6)])
    g = gate_durable(wf, thr=FAST)
    assert g.passed is True
    assert {"wfe_aggregate", "fold_dispersion"} <= {c.name for c in g.checks}


def test_durable_rejects_wfe_too_high():
    wf = _fake_wf([(1.5, [3, 3, -1, -1] * 6), (2.0, [3, 3, 3, -1] * 6)])
    g = gate_durable(wf, thr=FAST)
    assert g.passed is False                         # "too good to be true" rejects
    assert "wfe_aggregate" in {c.name for c in g.failed_checks}


# ── GENERAL ───────────────────────────────────────────────────────────────────
def test_general_cross_market_reality_check():
    rng = np.random.default_rng(2)
    logs = {f"m{i}": TradeLog.from_arrays(r=rng.normal(0, 1, 120)) for i in range(6)}
    logs["winner"] = TradeLog.from_arrays(r=rng.normal(0.8, 1.0, 120))
    g = gate_general(logs, thr=FAST)
    chk = next(c for c in g.checks if c.name == "cross_market_reality_check")
    assert 0.0 < chk.value <= 1.0


# ── run_gauntlet composition ──────────────────────────────────────────────────
def test_run_gauntlet_runs_real_and_strong_only_by_default():
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    gaunt = run_gauntlet(tl, thr=FAST)
    assert [g.name for g in gaunt.gates] == ["REAL", "STRONG"]
    assert gaunt.passed is True


def test_run_gauntlet_adds_durable_and_general_when_supplied():
    tl = TradeLog.from_arrays(r=[3.0, 3.0, 3.0, -1.0] * 40)
    wf = _fake_wf([(0.6, [3, 3, -1, -1] * 6), (0.7, [3, 3, 3, -1] * 6)])
    logs = {"a": tl, "b": TradeLog.from_arrays(r=[2.0, 2.0, -1.0, -1.0] * 30)}
    gaunt = run_gauntlet(tl, wf=wf, trade_logs=logs, thr=FAST)
    assert [g.name for g in gaunt.gates] == ["REAL", "STRONG", "DURABLE", "GENERAL"]
