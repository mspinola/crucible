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
from typing import Dict, Tuple, Union

import pandas as pd

from crucible.edge.metrics import expectancy
from crucible.edge.stats import Metric, Verdict, reality_check
from crucible.edge.trade_log import TradeLog

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


def full_sample(trades: TradeLog, metric: Metric = expectancy,
                metric_name: str = "expectancy", n_boot: int = 10_000,
                seed: int = 0) -> Verdict:
    """The whole-book verdict — the *full-range* counterpart to :func:`holdout`.

    No split: measure the edge across the entire log, in-sample. Useful for a
    "does an edge exist here at all, using every trade" read, but it is NOT an
    honest out-of-sample test — nothing is held back, so a positive verdict here
    can still be overfit. Prefer :func:`holdout` (or :func:`segmented_holdout`)
    when you need the honest read. Thin wrapper over
    :func:`crucible.edge.stats.reality_check` that names the intent."""
    return reality_check(trades, metric, metric_name, n_boot, seed=seed)


@dataclass
class SegmentedHoldout:
    """One holdout, read overall AND sliced by a grouping column.

    `overall` is the whole-book holdout; `segments` maps each distinct value of
    the grouping column to its own :class:`HoldoutResult` — every segment split
    at the same date with the same leakage control, so a per-segment TEST
    verdict is directly comparable to the overall one (and to the other
    segments). Thin segments are kept, not dropped: a segment whose TEST side is
    below `min_n` is flagged (``thin`` / :meth:`reliable`) rather than hidden, so
    the caller decides what to trust.
    """
    split: pd.Timestamp
    embargo_weeks: int
    by: str
    overall: HoldoutResult
    segments: Dict[str, HoldoutResult]
    min_n: int = 0

    def reliable(self) -> Dict[str, HoldoutResult]:
        """Segments whose TEST side clears `min_n` (all of them when min_n=0)."""
        return {k: v for k, v in self.segments.items() if v.test_n >= self.min_n}

    def thin(self) -> Dict[str, HoldoutResult]:
        """Segments whose TEST side is below `min_n` — kept but flagged."""
        return {k: v for k, v in self.segments.items() if v.test_n < self.min_n}

    def __str__(self) -> str:
        head = (f"SEGMENTED HOLDOUT @ {self.split.date()} by '{self.by}' "
                f"(embargo {self.embargo_weeks}w)")
        lines = [head, f"  OVERALL  n={self.overall.test_n:<5}"
                       f"{HoldoutResult._line(self.overall.test)}"]
        for name, res in self.segments.items():
            flag = "  (thin)" if res.test_n < self.min_n else ""
            lines.append(f"  {name:<8} n={res.test_n:<5}"
                         f"{HoldoutResult._line(res.test)}{flag}")
        return "\n".join(lines)


def segmented_holdout(trades: TradeLog, by: str, split: DateLike,
                      embargo_weeks: int = 8, metric: Metric = expectancy,
                      metric_name: str = "expectancy", n_boot: int = 10_000,
                      seed: int = 0, min_n: int = 0) -> SegmentedHoldout:
    """Run :func:`holdout` on the whole log and, separately, on each slice of it
    defined by the `by` column (e.g. ``by="asset_class"`` or ``by="symbol"``).

    Same split date, embargo, and leakage control for every segment, so the
    per-segment TEST verdicts line up with the overall one — the numbers a
    :func:`crucible.report.segment_forest` reads to draw a per-segment CI forest.
    Composes on a single grouping axis; cross a second axis (e.g. long vs short)
    by calling this on the already-filtered log, so the primitive stays small."""
    if by not in trades.frame.columns:
        raise ValueError(
            f"segmented_holdout needs a '{by}' column to group on. "
            f"Got: {list(trades.frame.columns)}"
        )
    overall = holdout(trades, split, embargo_weeks, metric, metric_name,
                      n_boot, seed=seed)
    segments: Dict[str, HoldoutResult] = {}
    col = trades.frame[by]
    for value in sorted(col.dropna().unique(), key=str):
        sub = TradeLog(trades.frame[col == value].reset_index(drop=True))
        segments[str(value)] = holdout(sub, split, embargo_weeks, metric,
                                       metric_name, n_boot, seed=seed)
    return SegmentedHoldout(
        split=pd.Timestamp(split), embargo_weeks=embargo_weeks, by=by,
        overall=overall, segments=segments, min_n=min_n,
    )
