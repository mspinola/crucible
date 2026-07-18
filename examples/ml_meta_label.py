"""Worked example for the tutorial (docs/tutorial.md §13; https://mspinola.github.io/crucible/tutorial/).

A meta-labeling take/skip filter, run end to end through crucible's ML track and
gauntlet. A primary signal has already produced a book of candidate trades; a model
scores each one (take or skip). The questions §13 walks through:

  1. Is the score real?      information_coefficient + alpha_gate + fold_ic
  2. Does it rank outcomes?  quantile_decay (win rate should climb Q1 -> Q5)
  3. Does the filter pay?    run_gauntlet on the filtered book vs the unfiltered one

Uses a reproducible synthetic book (no network) so it prints the exact numbers the
tutorial walks through. Each candidate trade carries a hidden edge *quality*; the
model's score reads that quality imperfectly (a real but noisy signal, not the
label itself), and the realized R-multiple is driven by the same quality — so a
higher score genuinely selects better trades, without the score ever seeing the
outcome.

    python examples/ml_meta_label.py
"""
import numpy as np
import pandas as pd

from crucible.edge import TradeLog
from crucible.ml import information_coefficient, alpha_gate, quantile_decay, fold_ic
from crucible.ml.ic import AlphaGateError
from crucible.validation import run_gauntlet


def synthetic_meta_book(n: int = 1600, seed: int = 3) -> pd.DataFrame:
    """A book of `n` candidate trades: a hidden quality drives both the model's
    score and the realized outcome, so the score has genuine (noisy) skill.

    Columns: `score` (the model's take/skip confidence), `label` (1 win / 0 loss),
    `r` (R-multiple: a winner banks +2R at its target, a loser -1R at its stop,
    both with mild fill noise).

    The parameters are set so the primary book is a *marginal* edge — real but not
    strong — and the top-score trades are clearly strong, so the filter demonstrably
    pays. That qualitative outcome holds across most seeds (~24/30); this one is
    representative, not cherry-picked."""
    rng = np.random.default_rng(seed)
    quality = rng.normal(0.0, 1.0, n)                 # hidden per-trade edge quality
    score = quality + rng.normal(0.0, 1.5, n)         # model reads it imperfectly
    p_win = 1.0 / (1.0 + np.exp(-(-0.50 + 0.85 * quality)))  # quality lifts win prob
    win = rng.random(n) < p_win
    r = np.where(win, 2.0 + rng.normal(0, 0.30, n), -1.0 + rng.normal(0, 0.15, n))
    return pd.DataFrame({"score": score, "label": win.astype(int), "r": r})


def _verdict(g) -> str:
    real = next((gate for gate in g.gates if gate.name == "REAL"), None)
    strong = next((gate for gate in g.gates if gate.name == "STRONG"), None)
    return (f"REAL {'PASS' if real and real.passed else 'FAIL'} · "
            f"STRONG {'PASS' if strong and strong.passed else 'FAIL'} · "
            f"gauntlet {'PASS' if g.passed else 'FAIL'}")


def main() -> None:
    book = synthetic_meta_book()
    preds = book[["score", "label"]]

    print(f"candidate trades: {len(book)}")
    print()

    # 1. Is the score real? -------------------------------------------------
    ic = information_coefficient(preds)
    print(f"[1] information coefficient  IC = {ic:+.4f}")
    try:
        alpha_gate(ic, min_ic=0.03)
        print(f"    alpha_gate(min_ic=0.03): PASS")
    except AlphaGateError as e:
        print(f"    alpha_gate(min_ic=0.03): FAIL — {e}")
    folds = fold_ic(book.reset_index(drop=True), ["score"], target="label", n_splits=5)
    row = folds.iloc[0]
    print(f"    out-of-fold IC = {row['ic_mean']:+.4f} ± {row['ic_std']:.4f} "
          f"(IR {row['ic_ir']:+.2f}) over {int(row['n_folds'])} folds")
    print()

    # 2. Does it rank outcomes? --------------------------------------------
    decay = quantile_decay(preds, q=5)
    print(f"[2] quantile decay (win rate by score quantile):")
    for _, qr in decay.table.iterrows():
        bar = "#" * round(qr["win_rate"] * 40)
        print(f"    Q{int(qr['quantile'])}  {qr['win_rate']*100:5.1f}%  (n={int(qr['count'])})  {bar}")
    print(f"    monotonic: {decay.monotonic} · spread Q5-Q1 = {decay.spread*100:+.1f} pts")
    print()

    # 3. Does the filter pay? ----------------------------------------------
    take = book[book["score"] >= book["score"].quantile(0.60)]   # take top 40%
    def stats(df):
        r = df["r"].to_numpy()
        pf = r[r > 0].sum() / -r[r < 0].sum()
        return f"n={len(df):4d}  win={100*(r>0).mean():4.1f}%  E={r.mean():+.3f}R  PF={pf:.2f}"
    print(f"[3] filter takes the top 40% by score:")
    print(f"    unfiltered   {stats(book)}")
    print(f"    filtered     {stats(take)}")
    print()

    g_all = run_gauntlet(TradeLog.from_arrays(r=book["r"].to_numpy()), n_variants=1)
    g_take = run_gauntlet(TradeLog.from_arrays(r=take["r"].to_numpy()), n_variants=1)
    print(f"    unfiltered gauntlet:  {_verdict(g_all)}")
    print(f"    filtered   gauntlet:  {_verdict(g_take)}")


if __name__ == "__main__":
    main()
