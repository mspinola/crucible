from crucible.edge import TradeLog
from crucible.validation import walk_forward, WalkForwardResult
from crucible.strategies import ma_cross


def test_walk_forward_runs_and_stitches(ohlc):
    res = walk_forward(
        ohlc, ma_cross,
        param_grid={"fast": [10, 20], "slow": [50, 100]},
        side="long", is_days=365 * 2, oos_days=365,
        tp=2.0, sl=1.0, timeout=20, min_is_trades=3,
    )
    assert isinstance(res, WalkForwardResult)
    assert len(res.folds) >= 1
    # each fold picked a real param combo from the grid
    for f in res.folds:
        assert f.best_params["fast"] in (10, 20)
        assert f.best_params["slow"] in (50, 100)
    # stitched OOS is the concatenation of the per-fold OOS trade logs
    assert isinstance(res.stitched, TradeLog)
    assert res.stitched.n == sum(f.oos_trades.n for f in res.folds)
    assert "WALK-FORWARD" in str(res)


def test_oos_windows_are_disjoint_and_ordered(ohlc):
    res = walk_forward(ohlc, ma_cross, param_grid={"fast": [20], "slow": [50]},
                       is_days=365 * 2, oos_days=365, min_is_trades=1)
    starts = [f.oos_start for f in res.folds]
    assert starts == sorted(starts)
    # no overlap: each OOS window starts at/after the previous window's end
    for a, b in zip(res.folds, res.folds[1:]):
        assert b.oos_start >= a.oos_end
