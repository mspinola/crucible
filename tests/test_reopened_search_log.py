"""A re-opened `SearchSpaceLog` must never resolve to a search of 0.

The governed workflow runs the search and the judge in separate processes: one
writes the ledger to disk, the other re-opens it and hands it to the gate. The
re-opened object has recorded nothing itself, so its `session_n_variants` is 0
while the file holds the whole search. Before this was caught, `variant_count`
returned that 0 and `run_gauntlet(..., n_variants=log)` fell through to the
uncorrected p-value with no error: a gate with its multiple-testing correction
switched off. These tests pin the four cases that have to stay distinct.
"""
import numpy as np
import pytest

from crucible.edge import TradeLog
from crucible.validation import (
    SearchSpaceLog,
    deflated_expectancy,
    deflated_sharpe,
    run_gauntlet,
    sidak_correction,
    variant_count,
)

SCOPE = "ES:two_arm_study"


def _written_then_reopened(tmp_path, n=2):
    """Write `n` variants in one log, then open a second on the same file: what a
    second process sees."""
    path = str(tmp_path / "search_log.jsonl")
    writer = SearchSpaceLog(scope=SCOPE, path=path)
    for i in range(n):
        writer.record({"arm": i}, status="tried")
    return writer, SearchSpaceLog(scope=SCOPE, path=path)


def _trades():
    return TradeLog.from_arrays(r=np.random.default_rng(0).normal(0.3, 1.0, 300))


def _real_check_names(gauntlet):
    return {c.name: c for g in gauntlet.gates for c in g.checks}


# ── a fresh log, used in the process that wrote it: unchanged ────────────────

def test_a_fresh_log_resolves_to_what_it_recorded(tmp_path):
    writer, _ = _written_then_reopened(tmp_path)
    assert variant_count(writer) == 2
    assert variant_count(SearchSpaceLog(scope=SCOPE)) == 0      # in-memory, nothing yet
    checks = _real_check_names(run_gauntlet(_trades(), n_variants=writer))
    assert "corrected_pvalue" in checks
    assert "2 variants" in checks["corrected_pvalue"].detail


def test_a_rerun_on_a_persisted_log_still_counts_this_run_only(tmp_path):
    # the reason session_n_variants exists: re-running one search must not
    # compound its own penalty. That behaviour is untouched.
    _, second = _written_then_reopened(tmp_path)
    second.record({"arm": 0}, status="tried")
    second.record({"arm": 1}, status="tried")
    assert second.n_variants == 4
    assert variant_count(second) == 2


# ── the bug: a re-opened log that recorded nothing in this process ───────────

def test_a_reopened_log_holds_the_search_but_no_session_count(tmp_path):
    # the diagnosis, pinned: the count is on disk, the session count is 0
    _, reopened = _written_then_reopened(tmp_path)
    assert reopened.n_variants == 2
    assert reopened.session_n_variants == 0


def test_variant_count_refuses_a_reopened_log(tmp_path):
    _, reopened = _written_then_reopened(tmp_path)
    with pytest.raises(ValueError, match=r"re-opened from disk.*holds 2 variant.*log\.n_variants"):
        variant_count(reopened)


def test_run_gauntlet_refuses_a_reopened_log_rather_than_skip_the_correction(tmp_path):
    _, reopened = _written_then_reopened(tmp_path)
    with pytest.raises(ValueError, match="re-opened from disk"):
        run_gauntlet(_trades(), n_variants=reopened)
    # what the error tells the caller to pass gives the corrected check
    checks = _real_check_names(run_gauntlet(_trades(), n_variants=reopened.n_variants))
    assert "corrected_pvalue" in checks and "permutation_pvalue" not in checks


def test_every_route_that_takes_a_ledger_refuses_a_reopened_one(tmp_path):
    _, reopened = _written_then_reopened(tmp_path)
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="re-opened from disk"):
        sidak_correction(0.01, reopened)
    with pytest.raises(ValueError, match="re-opened from disk"):
        deflated_sharpe([0.1, 0.2], returns=rng.normal(0.1, 1, 200), n_trials=reopened)
    with pytest.raises(ValueError, match="re-opened from disk"):
        deflated_expectancy(rng.normal(0.1, 1, 200),
                            [rng.normal(0, 1, 200), rng.normal(0, 1, 200)], n_trials=reopened)


def test_another_scope_on_the_same_file_does_not_trip_the_refusal(tmp_path):
    # only entries under THIS scope are the log's history
    path = str(tmp_path / "search_log.jsonl")
    SearchSpaceLog(scope="other", path=path).record({"x": 1})
    assert variant_count(SearchSpaceLog(scope=SCOPE, path=path)) == 0


# ── an empty log ────────────────────────────────────────────────────────────

def test_an_empty_log_resolves_to_zero_and_is_reported_as_uncorrected(tmp_path):
    for empty in (SearchSpaceLog(scope=SCOPE),
                  SearchSpaceLog(scope=SCOPE, path=str(tmp_path / "never_written.jsonl"))):
        assert variant_count(empty) == 0
        checks = _real_check_names(run_gauntlet(_trades(), n_variants=empty))
        assert "uncorrected" in checks["permutation_pvalue"].detail
        with pytest.raises(ValueError, match="n_variants must be >= 1"):
            sidak_correction(0.01, empty)


# ── an integer passed directly ──────────────────────────────────────────────

def test_an_int_passes_straight_through(tmp_path):
    assert variant_count(7) == 7
    assert variant_count(0) == 0
    assert variant_count(np.int64(3)) == 3
    _, reopened = _written_then_reopened(tmp_path)
    by_int = _real_check_names(run_gauntlet(_trades(), n_variants=2))["corrected_pvalue"]
    by_count = _real_check_names(
        run_gauntlet(_trades(), n_variants=reopened.n_variants))["corrected_pvalue"]
    assert by_int.value == by_count.value
