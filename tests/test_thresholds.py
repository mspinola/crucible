from crucible.validation import Thresholds


def test_select_overfit_bars_present_with_defaults():
    # The SELECT/overfit bars live on the central, overridable Thresholds home,
    # not hardcoded in the optimizer's select step.
    thr = Thresholds()
    assert thr.max_pbo == 0.5
    assert thr.min_deflated_sharpe == 0.95


def test_select_overfit_bars_are_overridable():
    thr = Thresholds(max_pbo=0.4, min_deflated_sharpe=0.99)
    assert thr.max_pbo == 0.4
    assert thr.min_deflated_sharpe == 0.99
