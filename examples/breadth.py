"""How many *independent* bets does a book really hold? — `crucible.breadth`.

A book of N markets is rarely N independent bets: currencies move as one dollar
bloc, the rates complex moves together, grains move together. `effective_n` reads
that structure straight off the return-correlation matrix and reports N_eff — the
participation ratio of its eigenvalues — plus the factors (principal components)
so you can name them.

Runs on synthetic data, no network:

    python examples/breadth.py
"""
import numpy as np
import pandas as pd

from crucible.breadth import effective_n


# A synthetic futures book: three correlated blocs + one lone market. The book
# lists 12 markets, but by construction it carries ~4 independent bets.
CLASSES = {
    "EUR": "fx", "JPY": "fx", "GBP": "fx", "AUD": "fx", "CAD": "fx", "CHF": "fx",
    "ZN": "rates", "ZF": "rates", "ZT": "rates",
    "ZC": "grain", "ZW": "grain",
    "GC": "metal",
}


def make_return_panel(n_weeks: int = 800, seed: int = 7) -> pd.DataFrame:
    """A (weeks x markets) weekly-return panel with a dollar bloc, a rates bloc, a
    grains bloc, and a lone metal — each bloc driven by one shared factor."""
    rng = np.random.default_rng(seed)
    dollar = rng.normal(0, 0.01, n_weeks)   # shared factors
    rates = rng.normal(0, 0.01, n_weeks)
    grain = rng.normal(0, 0.01, n_weeks)
    factor = {"fx": dollar, "rates": rates, "grain": grain}

    cols = {}
    for sym, ac in CLASSES.items():
        if ac in factor:
            # tight bloc: mostly the shared factor + a little idiosyncratic noise
            cols[sym] = 0.9 * factor[ac] + rng.normal(0, 0.004, n_weeks)
        else:
            cols[sym] = rng.normal(0, 0.01, n_weeks)   # GC: its own bet
    idx = pd.date_range("2010-01-01", periods=n_weeks, freq="W")
    return pd.DataFrame(cols, index=idx)


def main():
    panel = make_return_panel()
    b = effective_n(panel)

    print(f"window {panel.index[0].date()}..{panel.index[-1].date()} "
          f"({len(panel)} weeks), {b.n_assets} markets")
    print(f"N_eff (independent bets) = {b.n_eff:.1f}  of {b.n_assets} markets "
          f"— {b.redundancy:.1f} markets per bet\n")

    print(f"{'PC':>3}{'var%':>7}{'cum%':>7}   top-loading markets (sign, class)")
    cum = 0.0
    for i in range(min(6, b.n_assets)):
        ve = 100 * b.eigenvalues[i] / b.n_assets      # eigenvalues sum to n_assets
        cum += ve
        load = b.loadings.iloc[:, i]
        top = load.reindex(load.abs().sort_values(ascending=False).index)[:4]
        desc = ", ".join(f"{s}{'+' if v > 0 else '-'}({CLASSES[s]})" for s, v in top.items())
        print(f"{i + 1:>3}{ve:>6.1f}%{cum:>6.1f}%   {desc}")

    print(f"\nThe 12-market book is really ~{b.n_eff:.0f} bets — the dollar, rates, and\n"
          f"grain blocs each count once, plus the lone metal. That smaller number is the\n"
          f"honest denominator when you ask whether an edge across the book is real.")


if __name__ == "__main__":
    main()
