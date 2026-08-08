"""RealTest -> crucible, end to end: read a RealTest trade export and put the
edge through crucible's capital-free honesty checks.

RealTest (a paid, portfolio-level backtester) runs a strategy and, with
``SaveTradesAs`` set, writes its trade log to CSV. crucible reads that CSV,
turns it into a `TradeLog`, and answers the one question RealTest's equity
curve can't: *is this edge distinguishable from noise, or a small-sample
artifact?*

The one modeling choice is the **1R denominator**. RealTest exports P&L in
percent (``PctGain``), not R, and there is no per-trade risk column, so you
declare what 1R was. Here 1R is the protective-stop distance the paired
``examples/realtest/ma_cross.rts`` uses (``--r-pct``, default 5%), so
``r = PctGain / R_PCT``.

    pip install "crucible-quant[report]"          # [report] only for the tearsheet
    python examples/realtest_ingest.py                        # the bundled sample
    python examples/realtest_ingest.py --csv my_trades.csv --r-pct 2.0
    python examples/realtest_ingest.py --no-tearsheet         # console only, no plotly
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crucible.edge import TradeLog, edge_report, reality_check

SAMPLE = Path(__file__).resolve().parent / "realtest" / "ma_cross_trades.csv"


def pct_to_float(s: pd.Series) -> pd.Series:
    """RealTest percent fields export as strings like ``"3.45%"`` (or plain
    numbers). Strip a trailing ``%`` if present and return floats."""
    return s.astype(str).str.rstrip("%").astype(float)


def load_realtest_export(csv: Path, r_pct: float) -> TradeLog:
    raw = pd.read_csv(csv)

    # One correlated book only: crucible's bootstrap assumes independent trades,
    # so if a multi-strategy run duplicated a fill across sub-strategies, collapse
    # it. (The sample is a single strategy, so this is a no-op here.)
    before = len(raw)
    raw = raw.drop_duplicates(subset=["Symbol", "DateIn", "DateOut", "PriceIn"])
    if len(raw) < before:
        print(f"note: dropped {before - len(raw)} duplicate rows (correlated fills)")

    # Percent -> R. This single division is the whole "conversion"; the honesty
    # of everything downstream rides on r_pct being the real per-trade risk.
    r = pct_to_float(raw["PctGain"]) / r_pct
    mfe = pct_to_float(raw["PctMFE"]) / r_pct
    mae = pct_to_float(raw["PctMAE"]) / r_pct

    return TradeLog.from_arrays(
        r=r.to_numpy(),
        mfe=mfe.to_numpy(),
        mae=mae.to_numpy(),
        bars_held=raw["Bars"].to_numpy(),
        entry_date=pd.to_datetime(raw["DateIn"]).to_numpy(),
        exit_date=pd.to_datetime(raw["DateOut"]).to_numpy(),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a RealTest trade export into crucible.")
    p.add_argument("--csv", type=Path, default=SAMPLE,
                   help="RealTest SaveTradesAs CSV (default: the bundled MA-cross sample)")
    p.add_argument("--r-pct", type=float, default=5.0,
                   help="1R as a %% move — the strategy's protective-stop distance (default: 5)")
    p.add_argument("--out", default="realtest_tearsheet.html", help="tearsheet output path")
    p.add_argument("--no-tearsheet", action="store_true", help="skip the HTML tearsheet")
    args = p.parse_args()

    trades = load_realtest_export(args.csv, args.r_pct)
    print(f"\nloaded {trades.n} trades from {args.csv.name}  (1R = {args.r_pct:g}%)\n")
    print(edge_report(trades))
    print()
    print(reality_check(trades))            # <-- the verdict RealTest's Monte Carlo doesn't give

    if not args.no_tearsheet:
        from crucible.report import tearsheet
        path = tearsheet(trades, args.out,
                         title="SPY 20/50 MA cross (from RealTest)",
                         subtitle=f"{trades.n} trades · 1R = {args.r_pct:g}% stop · imported via SaveTradesAs")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
