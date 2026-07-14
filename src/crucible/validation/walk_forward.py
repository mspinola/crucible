"""Pardo walk-forward analysis, generic and capital-free.

Optimize a strategy's parameters on an in-sample window, apply the winner to the
next (untouched) out-of-sample window, step forward, repeat — then stitch every
OOS slice into one trade log. If the stitched OOS edge survives `reality_check`,
the strategy generalized; if it dies, the in-sample result was curve-fit.

The strategy is any callable ``strategy(prices, **params) -> boolean entry
Series``; the example `crucible.strategies` all match. Barriers turn entries into
R-multiple trades via `crucible.edge.barrier_trades`.

Windows carry Pardo's hygiene: PURGE (drop the tail of the in-sample window so a
trade straddling the IS/OOS boundary can't leak) and EMBARGO (skip the start of
each OOS window). Per fold we report Walk-Forward Efficiency (annualized OOS
return / annualized IS return) — >0.5-ish means OOS kept most of the IS edge.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from crucible.edge.trade_log import TradeLog
from crucible.edge.simulator import barrier_trades
from crucible.edge.metrics import expectancy

Strategy = Callable[..., pd.Series]
Score = Callable[[TradeLog], float]


def _default_score(tl: TradeLog) -> float:
    return expectancy(tl.r)


def _annualized(tl: TradeLog, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Total R over the WINDOW's calendar span in years — Pardo annualizes by the
    window, not the trades, so a sparse window can't blow up the denominator."""
    if tl.n == 0:
        return 0.0
    years = max((end - start).days / 365.25, 1e-9)
    return float(np.sum(tl.r)) / years


def _wfe(oos_tl, is_tl, oos_start, oos_end, is_start, is_end) -> float:
    """Walk-Forward Efficiency = annualized OOS / annualized IS. 0.0 when the IS
    baseline is non-positive (WFE is undefined against a losing in-sample, and
    0.0 fails low, which is the right direction)."""
    is_ann = _annualized(is_tl, is_start, is_end)
    if is_ann <= 0:
        return 0.0
    return _annualized(oos_tl, oos_start, oos_end) / is_ann


def _slice_by_entry(tl: TradeLog, lo: pd.Timestamp, hi: pd.Timestamp) -> TradeLog:
    e = pd.to_datetime(tl.frame["entry_date"])
    return TradeLog(tl.frame[(e >= lo) & (e < hi)].reset_index(drop=True))


@dataclass
class Fold:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: dict
    is_score: float
    oos_score: float
    wfe: float
    oos_trades: TradeLog


@dataclass
class WalkForwardResult:
    folds: List[Fold]
    stitched: TradeLog
    param_grid: dict = field(default_factory=dict)

    @property
    def mean_wfe(self) -> float:
        w = [f.wfe for f in self.folds]
        return float(np.mean(w)) if w else float("nan")

    def __str__(self) -> str:
        L = ["=" * 72, " WALK-FORWARD  (per fold: IS -> OOS, best params)", "=" * 72,
             f"{'OOS window':<26}{'IS E':>8}{'OOS E':>9}{'WFE':>8}   params"]
        for f in self.folds:
            window = f"{f.oos_start.date()}..{f.oos_end.date()}"
            L.append(f"{window:<26}{f.is_score:>+8.3f}{f.oos_score:>+9.3f}"
                     f"{f.wfe:>8.2f}   {f.best_params}")
        L.append("-" * 72)
        L.append(f"folds={len(self.folds)}  mean WFE={self.mean_wfe:.2f}  "
                 f"stitched OOS trades={self.stitched.n}")
        return "\n".join(L)


def _grid(param_grid: Dict[str, Sequence]) -> List[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]


def walk_forward(prices: pd.DataFrame, strategy: Strategy,
                 param_grid: Optional[Dict[str, Sequence]] = None, *,
                 side: str = "long", is_days: int = 365 * 3, oos_days: int = 365,
                 anchored: bool = True, purge_days: int = 5, embargo_days: int = 5,
                 tp: float = 2.0, sl: float = 1.0, timeout: int = 20,
                 score: Score = _default_score, min_is_trades: int = 10,
                 ) -> WalkForwardResult:
    """Run anchored (expanding IS) or rolling (fixed-length IS) walk-forward.

    is_days / oos_days are calendar-day window lengths. For each combo the full
    trade log is simulated ONCE on the whole price series (so indicator warmup is
    handled naturally); windows then slice trades by entry date. The best combo
    per fold is the one maximizing `score` on the leak-free in-sample trades.
    """
    param_grid = param_grid or {}
    combos = _grid(param_grid)
    idx = pd.to_datetime(prices.index)
    start, end = idx.min(), idx.max()

    # Simulate each parameter combo once over the full series.
    full: Dict[int, TradeLog] = {}
    for k, combo in enumerate(combos):
        entries = strategy(prices, **combo)
        full[k] = barrier_trades(prices, entries, side=side, tp=tp, sl=sl, timeout=timeout)

    folds: List[Fold] = []
    stitched_frames = []
    oos_start = start + pd.Timedelta(days=is_days)
    while oos_start < end:
        oos_end = min(oos_start + pd.Timedelta(days=oos_days), end)
        is_end = oos_start - pd.Timedelta(days=purge_days)
        is_start = start if anchored else max(start, oos_start - pd.Timedelta(days=is_days))
        test_lo = oos_start + pd.Timedelta(days=embargo_days)

        # Pick the combo with the best leak-free in-sample score.
        best_k, best_is_score, best_is_tl = None, -np.inf, None
        for k in range(len(combos)):
            tl = full[k]
            e = pd.to_datetime(tl.frame["entry_date"])
            x = pd.to_datetime(tl.frame["exit_date"])
            is_tl = TradeLog(tl.frame[(e >= is_start) & (e < is_end) & (x < is_end)]
                             .reset_index(drop=True))
            if is_tl.n < min_is_trades:
                continue
            s = score(is_tl)
            if s > best_is_score:
                best_k, best_is_score, best_is_tl = k, s, is_tl

        oos_start = oos_end  # advance regardless; only record folds that qualified
        if best_k is None:
            continue

        oos_tl = _slice_by_entry(full[best_k], test_lo, oos_end)
        folds.append(Fold(
            is_start=is_start, is_end=is_end,
            oos_start=test_lo, oos_end=oos_end,
            best_params=combos[best_k],
            is_score=best_is_score, oos_score=score(oos_tl),
            wfe=_wfe(oos_tl, best_is_tl, test_lo, oos_end, is_start, is_end),
            oos_trades=oos_tl,
        ))
        if oos_tl.n:
            stitched_frames.append(oos_tl.frame)

    stitched = TradeLog(pd.concat(stitched_frames, ignore_index=True)) if stitched_frames \
        else TradeLog(pd.DataFrame(columns=["r", "entry_date", "exit_date"]))
    return WalkForwardResult(folds=folds, stitched=stitched, param_grid=param_grid)
