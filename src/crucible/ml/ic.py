"""crucible.ml.ic — how much predictive signal is in a score.

The Information Coefficient is the Spearman rank correlation between a continuous
score (an ML probability, a factor value) and the realized outcome it tried to
predict. Rank-based, so it doesn't care whether the score is calibrated or how the
label is encoded — only whether higher scores line up with better outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class AlphaGateError(Exception):
    """Raised when a signal's Information Coefficient falls below the gate."""


def information_coefficient(preds: pd.DataFrame, *, score: str = "score",
                            label: str = "label") -> float:
    """Spearman rank IC between a continuous ``score`` and the realized ``label``.

    Rank-based, so it is invariant to the label encoding (+1/-1 or 0/1) and to any
    monotonic transform of the score. NaN/inf-safe. Returns ``0.0`` when a column
    is missing or fewer than five valid rows remain — a missing signal reads as no
    signal, never an exception.
    """
    if score not in preds.columns or label not in preds.columns:
        return 0.0
    sub = preds[[score, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return 0.0
    # Spearman == Pearson of the ranks; computing it this way keeps crucible.ml on
    # numpy/pandas only (pandas' method="spearman" would pull in scipy).
    ic = sub[score].rank().corr(sub[label].rank())
    return 0.0 if pd.isna(ic) else float(ic)


def alpha_gate(ic: float, *, min_ic: float) -> None:
    """Raise :class:`AlphaGateError` if ``ic`` is below ``min_ic``.

    A PASS/FAIL gate in crucible's idiom — wire it into a training loop to stop an
    edge-less or leaking model before it reaches a backtester.
    """
    if ic < min_ic:
        raise AlphaGateError(f"IC {ic:.4f} below gate {min_ic:.4f}")
