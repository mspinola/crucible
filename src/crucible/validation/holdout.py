"""Early-train / late-confirm temporal holdout.

The honest test for a FIXED strategy (nothing fitted): measure the edge on an
early period, freeze, then confirm it still holds on a late period the analysis
never touched. Leakage control matches the framework's `early_late_holdout`:

  * a train trade must have both ENTERED and EXITED before the split (so a trade
    whose forward window crosses the boundary can't leak into training), and
  * the first `embargo_weeks` of the test period are dropped.

Operates purely on a TradeLog carrying `entry_date` and `exit_date`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import pandas as pd

from crucible.edge.trade_log import TradeLog
from crucible.edge.metrics import expectancy
from crucible.edge.stats import reality_check, Verdict, Metric

DateLike = Union[str, pd.Timestamp]


def _need_dates(trades: TradeLog) -> None:
    for c in ("entry_date", "exit_date"):
        if c not in trades.frame.columns:
            raise ValueError(
                f"holdout needs a '{c}' column on the TradeLog (barrier_trades "
                f"provides it). Got: {list(trades.frame.columns)}"
            )


def split_train_test(trades: TradeLog, split: DateLike,
                     embargo_weeks: int = 8) -> Tuple[TradeLog, TradeLog]:
    """Leakage-controlled split. Train = entered AND exited before `split`;
    test = entered at least `embargo_weeks` after `split`."""
    _need_dates(trades)
    split = pd.Timestamp(split)
    test_start = split + pd.Timedelta(weeks=embargo_weeks)
    f = trades.frame.copy()
    entry = pd.to_datetime(f["entry_date"])
    exit_ = pd.to_datetime(f["exit_date"])
    train = f[(entry < split) & (exit_ < split)]
    test = f[entry >= test_start]
    return TradeLog(train.reset_index(drop=True)), TradeLog(test.reset_index(drop=True))


@dataclass
class HoldoutResult:
    split: pd.Timestamp
    embargo_weeks: int
    train: Verdict
    test: Verdict           # the verdict that matters — the untouched late period
    train_n: int
    test_n: int

    @property
    def label(self) -> str:
        return self.test.label

    def __str__(self) -> str:
        return (
            f"HOLDOUT @ {self.split.date()} (embargo {self.embargo_weeks}w)\n"
            f"  TRAIN  n={self.train_n:<5}{self._line(self.train)}\n"
            f"  TEST   n={self.test_n:<5}{self._line(self.test)}\n"
            f"  -> {self.label}  (verdict = the untouched TEST period)"
        )

    @staticmethod
    def _line(v: Verdict) -> str:
        return (f"E={v.point:+.3f}R  CI[{v.ci.low:+.3f},{v.ci.high:+.3f}]  "
                f"p(edge>0)={1 - v.p_value:.3f}  [{v.label}]")


def holdout(trades: TradeLog, split: DateLike, embargo_weeks: int = 8,
            metric: Metric = expectancy, metric_name: str = "expectancy",
            n_boot: int = 10_000, seed: int = 0) -> HoldoutResult:
    """Split, then run a full reality_check on each side. The TEST verdict is the
    honest read; the TRAIN verdict is context (it should look good — that's where
    an edge would have been chosen)."""
    train, test = split_train_test(trades, split, embargo_weeks)
    return HoldoutResult(
        split=pd.Timestamp(split),
        embargo_weeks=embargo_weeks,
        train=reality_check(train, metric, metric_name, n_boot, seed=seed),
        test=reality_check(test, metric, metric_name, n_boot, seed=seed),
        train_n=train.n,
        test_n=test.n,
    )
