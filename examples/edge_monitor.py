"""Watch a promoted edge for decay: freeze a baseline, then monitor against it.

The gauntlet judges a book once, before capital. This is the question that comes
after: is the edge still delivering what it promised? Synthetic book, no network,
seeded, so the printed numbers reproduce exactly.

    python examples/edge_monitor.py
"""
import numpy as np
import pandas as pd

from crucible.edge import TradeLog
from crucible.validation import (
    EdgeBaseline,
    Thresholds,
    cusum_design,
    edge_monitor,
    empirical_arl,
    rolling_expectancy,
)

# One trade shape reused throughout: 43% win rate, small losses, larger wins.
# Positive expectancy, strongly right-skewed, nothing like a Gaussian.
WIN_RATE, WIN_R, LOSS_R = 0.43, 1.75, -0.95


def synthetic_book(n, *, edge=1.0, seed=0, start="2015-01-01",
                   per_year=150.0) -> TradeLog:
    """A book of `n` closed trades in R. `edge` scales the per-trade mean without
    touching the shape, so `edge=0.5` is an edge that halved and nothing else."""
    rng = np.random.default_rng(seed)
    win = rng.random(n) < WIN_RATE
    r = np.where(win, rng.normal(WIN_R, 0.7, n), rng.normal(LOSS_R, 0.3, n))
    r = r - r.mean() + r.mean() * edge
    dates = pd.date_range(start, periods=n, freq=f"{365.25 / per_year:.3f}D")
    return TradeLog.from_arrays(r, entry_date=dates, exit_date=dates)


def thin(trades: TradeLog, keep_every: int) -> TradeLog:
    """Same per-trade distribution, same calendar span, fewer trades. Models a
    signal that stops firing without its per-trade edge changing at all."""
    return TradeLog(trades.frame.iloc[::keep_every].reset_index(drop=True))


def main():
    # ── 1. the validated book, and the baseline frozen from it ──────────────────
    validated = synthetic_book(2000, seed=1, start="2015-01-01")
    raw_mean = float(validated.r.mean())

    # The number the parameters were optimized on is biased high. Pretend the
    # search correction knocked 20% off it; anchoring to the corrected figure is
    # the whole reason to run this in crucible rather than by hand.
    honest = EdgeBaseline.from_log(validated, deflated_expectancy=raw_mean * 0.8,
                                   n_variants=64)
    naive = EdgeBaseline.from_log(validated)          # deflated=False, and it says so

    print("1) THE FROZEN BASELINE")
    print(f"   in-sample mean      {raw_mean:+.4f}R over {validated.n} trades")
    print(f"   deflated baseline   {honest.expectancy:+.4f}R   deflated={honest.deflated}")
    print(f"   naive baseline      {naive.expectancy:+.4f}R   deflated={naive.deflated}")
    print(f"   firing rate         {honest.trades_per_year:.0f} trades/yr")
    print("   The naive baseline is 25% higher, so every ratio measured against it")
    print("   is flattered by the same amount. Build this ONCE, at promotion.")

    # ── 2. the detector, designed rather than typed in ──────────────────────────
    design = cusum_design(honest)
    print("\n2) THE DETECTOR")
    print("  ", str(design).replace("\n", "\n   "))

    # The design's ARLs assume Gaussian increments. Trade R is not Gaussian, so
    # check rather than assume. Resample the book's own returns.
    print("\n3) CHECKING THE DESIGN'S CLAIMS AGAINST THIS BOOK'S OWN RETURNS")
    for shift, label in ((1.0, "in control"), (design.detect_shift, "edge halved")):
        e = empirical_arl(design, validated.r, baseline_expectancy=honest.expectancy,
                          shift=shift, n_sims=300, seed=7)
        print(f"   {label:<12} {str(e).splitlines()[0]}")
        print(f"                {str(e).splitlines()[1].strip()}")

    # ── 3. three live books ─────────────────────────────────────────────────────
    healthy = synthetic_book(1300, edge=1.0, seed=2, start="2022-01-01")
    halved = synthetic_book(1300, edge=0.5, seed=3, start="2022-01-01")
    drying = thin(healthy, 3)          # identical trade shape, a third of the rate

    print("\n4) THREE LIVE BOOKS, SAME FROZEN BASELINE")
    for name, live in (("edge intact", healthy), ("edge halved", halved),
                       ("signal drying up", drying)):
        v = edge_monitor(live, honest)
        print(f"\n   --- {name} ---")
        print("  ", str(v).replace("\n", "\n   "))

    # ── 4. why only the calibrated detector may escalate ────────────────────────
    v = edge_monitor(healthy, honest)
    roll = rolling_expectancy(healthy, v.window).dropna()
    below = float((roll < honest.expectancy * Thresholds().monitor_slip_ratio).mean())

    print("\n5) THE REASON THE SOFT CHANNEL CANNOT SAY 'DEGRADED'")
    print("   The 'edge intact' book was generated with edge=1.0, so its TRUE per-trade")
    print(f"   edge is {raw_mean:+.4f}R, a quarter ABOVE the {honest.expectancy:+.4f}R "
          "baseline. It never decayed.")
    print(f"   Even so, its trailing {v.window}-trade read dips below the 50% line in "
          f"{below:.0%} of windows,")
    print(f"   ranging {roll.min() / honest.expectancy:.0%} to "
          f"{roll.max() / honest.expectancy:.0%} of baseline on noise alone.")
    print("   A rule with no stated false-alarm rate would have cut a healthy book on")
    print(f"   whichever of those windows you happened to read. (Today's is "
          f"{v.edge_ratio:.0%}, which")
    print("   would have looked reassuring, and that is the same coin landing the other")
    print("   way up.)")
    print(f"   The CUSUM, which does have a stated rate, peaked at "
          f"{v.cusum_peak / v.cusum_h:.0%} of threshold and stayed silent.")


if __name__ == "__main__":
    main()
