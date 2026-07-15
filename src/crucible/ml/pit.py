"""crucible.ml.pit — point-in-time windows that can't peek past the decision bar.

The one discipline that keeps a feature honest: at decision time you may only read
bars that have already closed. These two helpers slice a time-indexed frame to the
``lookback`` rows up to a cutoff:

  window_before   excludes the cutoff bar — the training side, where an entry fills
                  at the cutoff bar, so that bar's own values are unknowable.
  asof_window     includes the cutoff bar — the serving side, where the cutoff bar
                  has closed and the entry is the next bar.

Same construction, so a live feature is built identically to its training twin.
Assumes the frame is sorted by its (datetime) index.
"""
from __future__ import annotations

import pandas as pd


def window_before(frame, before, lookback: int = 5):
    """The ``lookback`` rows STRICTLY BEFORE ``before`` — ending at the last bar that
    closed before the cutoff, excluding the cutoff bar and everything after it.

    The training-side point-in-time window: if an entry fills at ``before``, that
    bar's own values and all later bars are unknowable at the decision and must
    never enter a feature. Returns an empty slice if the frame is empty or ``before``
    is NaT.
    """
    if len(frame) == 0 or pd.isna(before):
        return frame.iloc[0:0]
    pos = int(frame.index.searchsorted(pd.Timestamp(before), side="left"))  # first row >= before
    return frame.iloc[max(0, pos - lookback):pos]


def asof_window(frame, asof, lookback: int = 5):
    """The ``lookback`` rows ON OR BEFORE ``asof`` — the serving-side twin of
    :func:`window_before`. At decision time the ``asof`` bar has closed (known) and
    the entry is the NEXT bar, so including it mirrors training's "window ends at the
    trigger bar, before entry" exactly. Empty slice if the frame is empty or ``asof``
    is NaT.
    """
    if len(frame) == 0 or pd.isna(asof):
        return frame.iloc[0:0]
    pos = int(frame.index.searchsorted(pd.Timestamp(asof), side="right"))  # first row > asof
    return frame.iloc[max(0, pos - lookback):pos]
