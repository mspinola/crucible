"""Narrative guard for examples/stridsman_postpub.py.

The example's closing text asserts a story: the full-history scorecard looks fine
(the synthetic trends were planted for the rules to find) and the post-publication
gauntlet refuses the system. CI runs the example, which only proves it executes; a
change to the simulator or the gates that flipped either half would print a false
lesson with a green build. This pins the two facts the story depends on, through
the example's own `judge()` code path (not a reconstruction of it), without pinning
any of the report formatting.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crucible.edge import expectancy  # noqa: E402
from examples.stridsman_postpub import PUB_DATE, judge, synthetic_prices  # noqa: E402


def test_story_full_history_flatters_and_post_publication_fails():
    all_trades, post, gauntlet = judge(synthetic_prices())

    # enough trades on both sides of the boundary for the story to be about
    # evidence rather than a handful of fills
    assert all_trades.n >= 50
    assert post.n >= 50
    assert (post.frame["entry_date"] >= PUB_DATE).all()

    # the era the rules were built for looks good on its own scorecard...
    assert expectancy(all_trades.r) > 0

    # ...and the years after publication do not survive the gate
    assert not gauntlet.passed
    assert "REAL" in {g.name for g in gauntlet.failed_gates}


def test_example_is_deterministic():
    a = judge(synthetic_prices())
    b = judge(synthetic_prices())
    assert a[1].n == b[1].n
    assert a[2].passed == b[2].passed
    assert list(a[1].r) == list(b[1].r)
