"""Windowed segment analysis — where and when an edge lived.

A *descriptive* read of an existing TradeLog, not a fit/validate loop (that is
:func:`crucible.validation.walk_forward`, which refits per fold). Here nothing is
fitted: the closed trades are sliced into consecutive calendar windows and a
metric is measured per (segment, window), where segments are the distinct values
of a grouping column plus an aggregate ``OVERALL`` row.

The right resolution for a pooled book — per-trade is too noisy, one lifetime
number hides regime shifts; a coarse (segment × era) grid shows whether the edge
was steady or lived in one window. Cells below ``min_n`` are flagged, not hidden.

Operates on a TradeLog carrying ``entry_date``; groups on any extra column
(``asset_class``, ``symbol``, ``side`` ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from crucible.edge.metrics import expectancy
from crucible.edge.stats import Metric
from crucible.edge.trade_log import TradeLog

DateLike = Union[str, pd.Timestamp]

OVERALL = "OVERALL"


@dataclass
class WindowCell:
    n: int
    value: float          # the metric over this (segment, window); NaN when empty
    ok: bool              # n >= min_n — enough trades to read the cell honestly


@dataclass
class WindowedSegments:
    """A (segment × window) grid of a metric. ``rows`` maps each segment to one
    :class:`WindowCell` per window, in ``windows`` order; the ``OVERALL`` row
    (all segments pooled) comes first."""
    windows: List[Tuple[str, str]]        # [(y0, y1), ...] calendar-year edges
    rows: Dict[str, List[WindowCell]]
    by: str
    window_years: int
    min_n: int
    n_trades: int
    metric_name: str

    def __str__(self) -> str:
        head = (f"WINDOWED SEGMENTS by '{self.by}' "
                f"({self.window_years}y windows, metric={self.metric_name})")
        hdr = "  " + " ".join(f"{a}-{b}".center(11) for a, b in self.windows)
        lines = [head, " " * 10 + hdr]
        for seg, cells in self.rows.items():
            cellstr = " ".join(
                (f"{c.value:+.2f}({c.n})" if c.n else "  -  ").center(11)
                + ("" if c.ok or not c.n else "*")   # thin marker
                for c in cells
            )
            lines.append(f"{seg:<10}{cellstr}")
        return "\n".join(lines)


def _need_entry_date(trades: TradeLog) -> None:
    if "entry_date" not in trades.frame.columns:
        raise ValueError(
            "windowed_segments needs an 'entry_date' column on the TradeLog "
            f"(barrier_trades provides it). Got: {list(trades.frame.columns)}"
        )


def windowed_segments(trades: TradeLog, by: str, *, window_years: int = 4,
                      start: Optional[DateLike] = None,
                      end: Optional[DateLike] = None,
                      metric: Metric = expectancy,
                      metric_name: str = "expectancy",
                      min_n: int = 8) -> WindowedSegments:
    """Slice `trades` into consecutive `window_years`-year calendar windows and
    measure `metric` per (segment, window).

    Segments are the distinct values of the `by` column plus a pooled
    ``OVERALL`` row. Windows are aligned to the start year; `start`/`end` default
    to the log's own entry-date span. A cell with fewer than `min_n` trades is
    flagged (``ok=False``), not dropped — the caller decides what to trust."""
    _need_entry_date(trades)
    if by not in trades.frame.columns:
        raise ValueError(
            f"windowed_segments needs a '{by}' column to group on. "
            f"Got: {list(trades.frame.columns)}"
        )
    f = trades.frame
    entry = pd.to_datetime(f["entry_date"])
    if entry.empty:
        return WindowedSegments([], {OVERALL: []}, by, window_years, min_n, 0,
                                metric_name)
    y0 = pd.Timestamp(start).year if start is not None else int(entry.min().year)
    y1 = pd.Timestamp(end).year if end is not None else int(entry.max().year)
    # Enough whole windows (aligned to y0) for the last window's right edge to
    # sit strictly past y1, so a trade in year y1 always lands in a window.
    n_win = max(1, math.ceil((y1 - y0 + 1) / window_years))
    edges = [y0 + i * window_years for i in range(n_win + 1)]
    wins = [(pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{b}-01-01"))
            for a, b in zip(edges, edges[1:])]

    segments = [OVERALL] + sorted(f[by].dropna().unique(), key=str)
    rows: Dict[str, List[WindowCell]] = {}
    for seg in segments:
        mask = pd.Series(True, index=f.index) if seg == OVERALL else (f[by] == seg)
        r_seg = f.loc[mask, "r"].to_numpy(dtype=float)
        e_seg = entry[mask]
        cells: List[WindowCell] = []
        for a, b in wins:
            in_win = (e_seg >= a) & (e_seg < b)
            r = r_seg[in_win.to_numpy()]
            n = int(len(r))
            cells.append(WindowCell(n=n,
                                    value=float(metric(r)) if n else float("nan"),
                                    ok=n >= min_n))
        rows[str(seg)] = cells

    return WindowedSegments(
        windows=[(str(a.year), str(b.year)) for a, b in wins],
        rows=rows, by=by, window_years=window_years, min_n=min_n,
        n_trades=int(len(f)), metric_name=metric_name,
    )
