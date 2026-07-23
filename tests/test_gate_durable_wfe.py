import numpy as np

from crucible.edge import TradeLog
from crucible.validation import Thresholds, gate_durable


class _Fold:
    def __init__(self, is_score, oos_score, wfe, oos_returns):
        self.is_score, self.oos_score, self.wfe = is_score, oos_score, wfe
        self.oos_trades = TradeLog.from_arrays(r=np.asarray(oos_returns, dtype=float))


class _WF:
    def __init__(self, folds):
        self.folds = folds


def _folds(sqn_wfes, seed=0):
    rng = np.random.default_rng(seed)
    # is_score=1 so oos_score == the injected per-fold SQN-WFE ratio
    return _WF([_Fold(1.0, w, 0.6, rng.normal(0.5, 0.02, 30)) for w in sqn_wfes])


def _chk(gate, name):
    return next(c for c in gate.checks if c.name == name)


def test_sqn_wfe_aggregate_is_mean_of_fold_ratios():
    wfes = [0.8, 1.0, 1.2, 0.9]
    g = gate_durable(_folds(wfes), wfe="sqn")
    chk = _chk(g, "wfe_sqn_aggregate")
    assert chk.value == np.mean(wfes)
    assert chk.passed is True                       # 0.975 > 0.5


def test_sqn_wfe_fails_below_bar():
    g = gate_durable(_folds([0.2, 0.3, 0.1]), wfe="sqn")
    assert _chk(g, "wfe_sqn_aggregate").passed is False


def test_sqn_wfe_anomaly_is_soft():
    g = gate_durable(_folds([3.0, 4.0, 5.0]), wfe="sqn")   # far above 1
    anom = _chk(g, "wfe_sqn_not_anomalous")
    assert anom.hard is False and anom.passed is False     # flagged but doesn't gate
    assert _chk(g, "wfe_sqn_aggregate").passed is True      # hard check still passes


def test_default_is_return_wfe_unchanged():
    g = gate_durable(_folds([1.0, 1.0]), wfe="return", thr=Thresholds())
    names = {c.name for c in g.checks}
    assert "wfe_aggregate" in names and "wfe_sqn_aggregate" not in names


def test_custom_min_wfe_sqn_threshold():
    thr = Thresholds(min_wfe_sqn=1.5)
    g = gate_durable(_folds([1.0, 1.1, 0.9]), wfe="sqn", thr=thr)   # mean 1.0 < 1.5
    assert _chk(g, "wfe_sqn_aggregate").passed is False
