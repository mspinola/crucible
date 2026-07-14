"""The TradeLog contract — the one schema everything in crucible.edge speaks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Mapping, Sequence

import numpy as np
import pandas as pd

# Canonical schema. `r` is the only required column; the rest unlock more of the
# report (excursion metrics need mfe/mae, time metrics need bars_held, etc.).
REQUIRED = ("r",)
OPTIONAL = ("mfe", "mae", "bars_held", "prob", "entry_date", "exit_date")


@dataclass(frozen=True)
class TradeLog:
    """A set of closed trades, returns in a risk-normalized unit (R-multiples).

    Columns (canonical names):
      r          per-trade return in R (1R = the risk taken at entry)   [required]
      mfe, mae   max favorable / adverse excursion, in R                [excursion]
      bars_held  holding period, in bars                                [time]
      prob       model confidence at entry, 0..1                        [calibration]
      entry_date, exit_date                                             [ordering]

    Construct with :meth:`from_arrays` or :meth:`from_frame`. The frame is stored
    as-is (extra columns are allowed and preserved).
    """
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED if c not in self.frame.columns]
        if missing:
            raise ValueError(
                f"TradeLog is missing required column(s) {missing}. "
                f"Got columns: {list(self.frame.columns)}. "
                f"Use TradeLog.from_frame(df, mapping={{'your_col': 'r'}}) to rename."
            )

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def from_arrays(cls, r: Sequence[float], *, mfe=None, mae=None,
                    bars_held=None, prob=None, entry_date=None,
                    exit_date=None) -> "TradeLog":
        data = {"r": np.asarray(r, dtype=float)}
        for name, arr in (("mfe", mfe), ("mae", mae), ("bars_held", bars_held),
                          ("prob", prob), ("entry_date", entry_date),
                          ("exit_date", exit_date)):
            if arr is not None:
                data[name] = np.asarray(arr)
        return cls(pd.DataFrame(data))

    @classmethod
    def from_frame(cls, df: pd.DataFrame, *, r_col: str = "r",
                   mapping: Optional[Mapping[str, str]] = None) -> "TradeLog":
        """Wrap an existing frame. `mapping` renames your columns to the schema
        (e.g. ``{'pct_return': 'r'}``); `r_col` is a shorthand for the return
        column if you'd rather not build a full mapping."""
        rename = dict(mapping or {})
        if r_col != "r" and r_col in df.columns and "r" not in rename.values():
            rename[r_col] = "r"
        out = df.rename(columns=rename).copy()
        return cls(out)

    # ── accessors ────────────────────────────────────────────────────────────
    @property
    def r(self) -> np.ndarray:
        return self.frame["r"].to_numpy(dtype=float)

    def col(self, name: str) -> Optional[np.ndarray]:
        """Return a schema column as an array, or None if absent."""
        return self.frame[name].to_numpy() if name in self.frame.columns else None

    @property
    def n(self) -> int:
        return len(self.frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        cols = ", ".join(self.frame.columns)
        return f"TradeLog(n={self.n}, cols=[{cols}])"
