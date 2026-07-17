"""Reproducibility guard for the §13 ML worked example.

The tutorial quotes these exact numbers, so they must not drift silently — a
dependency bump that moves the IC or flips a gate should fail here, not surface
as a wrong number in the published page.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.ml_meta_label import synthetic_meta_book, main  # noqa: E402
from crucible.ml import information_coefficient, quantile_decay  # noqa: E402
from crucible.edge import TradeLog  # noqa: E402
from crucible.validation import run_gauntlet  # noqa: E402


def _gate(g, name):
    return next(x for x in g.gates if x.name == name).passed


def test_score_is_real_and_ranks_outcomes():
    book = synthetic_meta_book()
    assert len(book) == 1600
    preds = book[["score", "label"]]
    assert information_coefficient(preds) == pytest.approx(0.2178, abs=1e-3)
    decay = quantile_decay(preds, q=5)
    assert decay.monotonic
    wr = decay.table["win_rate"].to_numpy()
    assert wr[0] == pytest.approx(0.266, abs=2e-3)   # Q1
    assert wr[-1] == pytest.approx(0.578, abs=2e-3)  # Q5


def test_filter_lifts_a_marginal_book_to_strong():
    book = synthetic_meta_book()
    take = book[book["score"] >= book["score"].quantile(0.60)]
    assert len(take) == 640

    g_all = run_gauntlet(TradeLog.from_arrays(r=book["r"].to_numpy()), n_variants=1)
    g_take = run_gauntlet(TradeLog.from_arrays(r=take["r"].to_numpy()), n_variants=1)

    # the primary book is real but marginal — it fails STRONG on the PF CI lower
    # bound (point PF 1.33 clears 1.25, the lower bound does not)
    assert _gate(g_all, "REAL") and not _gate(g_all, "STRONG")
    # the top-score trades are clearly strong — the filter pays
    assert _gate(g_take, "REAL") and _gate(g_take, "STRONG")


def test_script_runs(capsys):
    main()
    out = capsys.readouterr().out
    assert "IC = +0.2178" in out
    assert "monotonic: True" in out
    assert "filtered   gauntlet:  REAL PASS · STRONG PASS · gauntlet PASS" in out
