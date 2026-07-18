"""Full-range run of the tutorial's Donchian breakout: the whole-history read.

The cheapest run mode. Describe the edge, then reality-check whether it clears zero
over the entire trade log at once. On this breakout the verdict is HELD: real in
aggregate, the picture a backtester would sell you. See donchian_gauntlet.py for why
the full gauntlet still rejects it, and the Run modes page for all three side by side:
https://mspinola.github.io/crucible/run_modes/

Reuses the exact prices and signal from donchian_gauntlet.py, so the numbers match
the tutorial and the site.

    pip install "crucible-quant[report]"
    python examples/donchian_fullrange.py     # prints the verdict, writes donchian_fullrange.html
"""
from crucible.edge import barrier_trades, edge_report, reality_check

from donchian_gauntlet import donchian, synthetic_prices


def main() -> None:
    px = synthetic_prices()
    trades = barrier_trades(px, donchian(px, 20), side="long", tp=2.5, sl=1.0, timeout=30)

    print(edge_report(trades))
    print("\n" + str(reality_check(trades)))       # the whole-history verdict (HELD)

    # the shareable scorecard (needs the [report] extra)
    from crucible.report import fullrange_scorecard
    path = fullrange_scorecard(
        trades, "donchian_fullrange.html", title="Donchian breakout",
        subtitle="20-bar high breakout · 2.5R / 1R / 30-bar cap")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
