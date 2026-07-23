import numpy as np
import pytest

from crucible.validation import deflated_sharpe, pbo_cscv
from crucible.validation.pbo import _sharpe_cols

# --------------------------------------------------------------------------- #
#  pbo_cscv
# --------------------------------------------------------------------------- #

def test_pbo_all_noise_centers_on_a_coin_flip():
    """N configs of pure i.i.d. noise: nothing to select, so the best-in-sample
    config is a coin flip out of sample. A SINGLE matrix gives a noisy PBO (it can
    land anywhere), so average over realizations — the center sits near 0.5."""
    pbos = []
    for seed in range(16):
        M = np.random.default_rng(seed).normal(0.0, 1.0, size=(240, 30))
        res = pbo_cscv(M, S=10)
        assert res.n_splits == 252        # C(10, 5)
        assert res.n_configs == 30
        pbos.append(res.pbo)
    assert 0.40 < np.mean(pbos) < 0.68    # coin-flip center (mild >0.5 discretization bias)


def test_pbo_persistent_edge_is_robust():
    """One column carries a real, time-stable edge among noise: it wins in-sample
    and keeps winning out of sample, so PBO collapses toward 0."""
    rng = np.random.default_rng(1)
    M = rng.normal(0.0, 1.0, size=(300, 24))
    M[:, 7] += 0.8                        # a genuinely superior config in every period
    res = pbo_cscv(M, S=12)
    assert res.pbo < 0.05
    assert res.oos_below_zero < 0.05
    assert res.label == "ROBUST"
    # It also decisively beats the noise center — the whole point of the tool.
    noise = pbo_cscv(rng.normal(0.0, 1.0, size=(300, 24)), S=12)
    assert res.pbo < noise.pbo


def test_pbo_rejects_odd_or_too_many_blocks():
    M = np.random.default_rng(2).normal(size=(100, 5))
    with pytest.raises(ValueError):
        pbo_cscv(M, S=7)                  # odd
    with pytest.raises(ValueError):
        pbo_cscv(M, S=200)                # more blocks than rows


def test_pbo_rejects_nonfinite_matrix():
    M = np.zeros((100, 4))
    M[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pbo_cscv(M, S=10)


def test_sharpe_cols_zero_dispersion_is_zero():
    block = np.tile([0.5, 0.0, -0.5], (8, 1))   # each column constant down its rows
    assert np.allclose(_sharpe_cols(block), 0.0)


# --------------------------------------------------------------------------- #
#  deflated_sharpe
# --------------------------------------------------------------------------- #

def test_deflated_sharpe_null_search_not_significant():
    """Best of many noise trials: its Sharpe is a selection artifact, so once
    deflated for the search size it should NOT read significant."""
    rng = np.random.default_rng(3)
    trials = [rng.normal(0.0, 1.0, 250) for _ in range(200)]
    sharpes = [t.mean() / t.std(ddof=1) for t in trials]
    winner = trials[int(np.argmax(sharpes))]
    res = deflated_sharpe(sharpes, returns=winner)
    assert res.deflated_sharpe < 0.95
    assert res.sr0_threshold > 0          # the bar rises with the trial count


def test_deflated_sharpe_real_edge_significant():
    """A genuinely strong Sharpe from a small search clears the deflated bar."""
    rng = np.random.default_rng(4)
    trials = [rng.normal(0.0, 1.0, 500) for _ in range(9)]
    winner = rng.normal(0.20, 1.0, 500)   # per-period Sharpe ~0.2 over 500 obs
    sharpes = [t.mean() / t.std(ddof=1) for t in trials] + [winner.mean() / winner.std(ddof=1)]
    res = deflated_sharpe(sharpes, returns=winner)
    assert res.deflated_sharpe > 0.95
    assert res.label == "SIGNIFICANT"


def test_deflated_sharpe_needs_enough_trials_and_obs():
    with pytest.raises(ValueError):
        deflated_sharpe([0.1], returns=np.random.default_rng(5).normal(size=100))
    with pytest.raises(ValueError):
        deflated_sharpe([0.1, 0.2, 0.3], returns=[0.5])


# ── n_trials: how many you TRIED vs how many you SCORED ──────────────────────

def _trials_and_returns(seed=0, k=3):
    rng = np.random.default_rng(seed)
    return rng.normal(0.01, 0.09, k), rng.normal(0.02, 0.1, 240)


def test_n_trials_defaults_to_the_number_of_trial_sharpes():
    tr, r = _trials_and_returns()
    assert deflated_sharpe(tr, returns=r).n_trials == len(tr)
    assert (deflated_sharpe(tr, returns=r).deflated_sharpe
            == deflated_sharpe(tr, returns=r, n_trials=len(tr)).deflated_sharpe)


def test_a_bigger_search_raises_the_bar_and_lowers_the_verdict():
    """The whole point: grow the search and the winner must clear more."""
    tr, r = _trials_and_returns()
    small = deflated_sharpe(tr, returns=r, n_trials=3)
    large = deflated_sharpe(tr, returns=r, n_trials=141)
    assert large.sr0_threshold > small.sr0_threshold
    assert large.deflated_sharpe <= small.deflated_sharpe
    assert large.n_trials == 141        # recorded, so the correction is auditable


def test_the_variance_estimate_still_comes_from_the_scored_trials():
    """n_trials moves the bar; it must not silently change the dispersion estimate."""
    tr, r = _trials_and_returns()
    wide = np.concatenate([tr, [0.5, -0.5]])
    assert (deflated_sharpe(wide, returns=r, n_trials=141).sr0_threshold
            > deflated_sharpe(tr, returns=r, n_trials=141).sr0_threshold)


def test_a_search_space_log_can_be_passed_directly():
    from crucible.validation import SearchSpaceLog
    log = SearchSpaceLog(scope="scan")
    for i in range(20):
        log.record({"i": i}, status="tried")
    tr, r = _trials_and_returns()
    assert deflated_sharpe(tr, returns=r, n_trials=log).n_trials == 20


def test_claiming_fewer_trials_than_were_scored_is_refused():
    tr, r = _trials_and_returns(k=5)
    with pytest.raises(ValueError, match="cannot have tried fewer configs than it scored"):
        deflated_sharpe(tr, returns=r, n_trials=2)


def test_too_few_scored_trials_still_names_the_variance_problem():
    _, r = _trials_and_returns()
    with pytest.raises(ValueError, match="estimate the search's variance"):
        deflated_sharpe([0.1], returns=r, n_trials=100)
