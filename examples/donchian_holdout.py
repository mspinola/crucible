"""Holdout run of the tutorial's Donchian breakout: an early-train / late-confirm split.

A stricter run mode than the full-range read. Fit on an early slice, confirm on a
later one the analysis never touched (with a purge/embargo band so a straddling trade
cannot leak across the split). The verdict is the untouched TEST period. This breakout
stays HELD across the split, yet still fails the full gauntlet (donchian_gauntlet.py)
on DURABLE. All three modes side by side on the Run modes page:
https://mspinola.github.io/crucible/run_modes/

Reuses the exact prices and signal from donchian_gauntlet.py, so the numbers match
the tutorial and the site.

    pip install "crucible-quant[report]"
    python examples/donchian_holdout.py       # prints the scorecard, writes donchian_holdout.html
"""
from crucible.edge import barrier_trades
from crucible.validation import holdout

from donchian_gauntlet import donchian, synthetic_prices

SPLIT = "2016-01-01"


def main() -> None:
    px = synthetic_prices()
    trades = barrier_trades(px, donchian(px, 20), side="long", tp=2.5, sl=1.0, timeout=30)

    print(holdout(trades, SPLIT, embargo_weeks=8))      # TRAIN vs the untouched TEST

    # the shareable scorecard (needs the [report] extra)
    from crucible.report import holdout_report
    path = holdout_report(trades, SPLIT, "donchian_holdout.html", embargo_weeks=8,
                          title="Donchian breakout")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
