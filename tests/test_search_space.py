"""SearchSpaceLog — the ledger behind an honest data-mining correction."""
import pytest

from crucible.validation import SearchSpaceLog


def test_counts_all_variants_including_discarded():
    # The whole point: a correction is only valid if it sees the variants you
    # THREW AWAY, not just the one you kept.
    log = SearchSpaceLog(scope='ES:ma_cross_grid')
    log.record({'fast': 10, 'slow': 50}, score=1.2, status='discarded')
    log.record({'fast': 20, 'slow': 50}, score=1.8, status='tried')
    log.mark_selected({'fast': 20, 'slow': 100}, score=2.1)
    assert log.n_variants == 3
    assert log.best()['params'] == {'fast': 20, 'slow': 100}


def test_rejects_bad_status():
    log = SearchSpaceLog(scope='ES:ma_cross_grid')
    with pytest.raises(ValueError):
        log.record({'fast': 10}, status='winner')


def test_in_memory_by_default_writes_nothing(tmp_path):
    # crucible core stays side-effect-free unless you ask for persistence.
    log = SearchSpaceLog(scope='ES:ma_cross_grid')
    log.record({'fast': 10, 'slow': 50})
    assert log.path is None
    assert not list(tmp_path.iterdir())


def test_persists_and_reloads(tmp_path):
    path = str(tmp_path / 'search_log.jsonl')
    log = SearchSpaceLog(scope='ES:ma_cross_grid', path=path)
    log.record({'fast': 10, 'slow': 50}, score=1.2, status='discarded')
    log.record({'fast': 20, 'slow': 50}, score=1.8, status='tried')

    # A later run accumulates onto the same log — counts survive restarts
    reloaded = SearchSpaceLog(scope='ES:ma_cross_grid', path=path)
    assert reloaded.n_variants == 2
    reloaded.record({'fast': 20, 'slow': 100}, score=2.1)
    assert reloaded.n_variants == 3


def test_session_n_variants_excludes_loaded_history(tmp_path):
    # Re-running the SAME search must not compound its own penalty just because
    # the ledger remembers the previous identical run. session_ is the honest N
    # for an idempotent re-run; n_variants is honest when each run explores anew.
    path = str(tmp_path / 'search_log.jsonl')
    first = SearchSpaceLog(scope='ES:ma_cross_grid', path=path)
    first.record({'fast': 10, 'slow': 50}, status='tried')
    first.record({'fast': 20, 'slow': 50}, status='tried')

    second = SearchSpaceLog(scope='ES:ma_cross_grid', path=path)  # reloads 2
    assert second.n_variants == 2            # total on record
    assert second.session_n_variants == 0    # nothing recorded THIS run yet
    second.record({'fast': 10, 'slow': 50}, status='tried')
    second.record({'fast': 20, 'slow': 50}, status='tried')
    assert second.n_variants == 4             # accumulates on disk
    assert second.session_n_variants == 2     # but this run only searched 2


def test_scope_isolation(tmp_path):
    # Two searches sharing one file must not contaminate each other's N.
    path = str(tmp_path / 'search_log.jsonl')
    es_log = SearchSpaceLog(scope='ES:params', path=path)
    es_log.record({'fast': 10}, score=1.0)
    gc_log = SearchSpaceLog(scope='GC:params', path=path)
    gc_log.record({'fast': 10}, score=1.5)

    assert SearchSpaceLog(scope='ES:params', path=path).n_variants == 1
    assert SearchSpaceLog(scope='GC:params', path=path).n_variants == 1


def test_scores_skips_unscored_entries():
    log = SearchSpaceLog(scope='ES:ma_cross_grid')
    log.record({'fast': 10}, score=1.0)
    log.record({'fast': 20})              # no score
    assert log.scores() == [1.0]


def test_best_is_none_when_nothing_scored():
    log = SearchSpaceLog(scope='ES:ma_cross_grid')
    log.record({'fast': 10})
    assert log.best() is None


def test_extra_kwargs_are_stored():
    # `fixed` params ride along as extra — the audit trail of what did NOT vary.
    log = SearchSpaceLog(scope='ES:ma_cross_grid')
    entry = log.record({'fast': 10}, fixed={'kind': 'sma', 'tp': 2.0})
    assert entry['fixed'] == {'kind': 'sma', 'tp': 2.0}
