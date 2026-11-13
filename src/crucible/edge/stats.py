"""The honesty layer — the reason crucible.edge exists.

A positive expectancy on a trade log means nothing until you know it could not
have come from noise. These tools answer that with a confidence interval, a
resampling p-value, and — when you have the price series — a random-entry null.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from crucible.edge.metrics import expectancy, profit_factor, sqn, win_rate
from crucible.edge.trade_log import TradeLog

Metric = Callable[[np.ndarray], float]

# The default metric set for `bootstrap_metric_cis` — the headline numbers a gate
# reasons over. Each is unit-consistent with the trade returns fed in (expectancy
# in R, profit_factor / win_rate ratios, sqn a t-like ratio), so the same call
# works on an R-multiple TradeLog.
_DEFAULT_METRICS: "dict[str, Metric]" = {
    "expectancy": expectancy,
    "profit_factor": profit_factor,
    "sqn": sqn,
    "win_rate": win_rate,
}


@dataclass
class CI:
    point: float
    low: float
    high: float
    alpha: float

    def __str__(self) -> str:
        lvl = int(round((1 - self.alpha) * 100))
        return f"{self.point:+.3f}  {lvl}% CI [{self.low:+.3f}, {self.high:+.3f}]"


@dataclass
class Verdict:
    metric: str
    point: float
    ci: CI
    p_value: float          # one-sided: P(metric <= 0) under resampling
    label: str              # "HELD" | "FRAGILE" | "FAIL"

    def __str__(self) -> str:
        lvl = int(round((1 - self.ci.alpha) * 100))
        note = {
            "HELD": "the edge clears zero across resamples.",
            "FRAGILE": "point positive, but the CI straddles zero — not "
                       "distinguishable from noise at this sample size. Do NOT size it up.",
            "FAIL": "no positive edge.",
        }[self.label]
        return (f"VERDICT ({self.metric}): {self.point:+.3f} R   "
                f"{lvl}% CI [{self.ci.low:+.3f}, {self.ci.high:+.3f}]\n"
                f"                     p(edge>0) = {1 - self.p_value:.3f}"
                f"        ->  {self.label}\n  {note}")


def _resample(r: np.ndarray, metric: Metric, n_boot: int, rng) -> np.ndarray:
    n = len(r)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = metric(rng.choice(r, size=n, replace=True))
    return draws


def bootstrap_ci(trades: TradeLog, metric: Metric = expectancy,
                 n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> CI:
    """Bootstrap confidence interval for any per-trade-return metric.

    The point estimate alone badly understates sampling noise at the trade
    counts real strategies produce (tens to low hundreds); the CI is the honest
    read. `metric` is any callable over the return array (expectancy,
    profit_factor, sqn, ...)."""
    r = trades.r
    if len(r) == 0:
        return CI(float("nan"), float("nan"), float("nan"), alpha)
    rng = np.random.default_rng(seed)
    draws = _resample(r, metric, n_boot, rng)
    finite = draws[np.isfinite(draws)]
    lo = float(np.percentile(finite, alpha / 2 * 100)) if len(finite) else float("nan")
    hi = float(np.percentile(finite, (1 - alpha / 2) * 100)) if len(finite) else float("nan")
    return CI(point=float(metric(r)), low=lo, high=hi, alpha=alpha)


def bootstrap_metric_cis(trades, metrics: "dict[str, Metric] | None" = None,
                         n_boot: int = 10_000, alpha: float = 0.05,
                         seed: int = 0) -> "dict[str, CI]":
    """Bootstrap CIs for a whole SET of metrics at once — the multi-metric
    companion to :func:`bootstrap_ci`.

    One resample loop, every metric read off each draw, returned as a
    ``{name: CI}`` mapping so a gate can compare each metric's CI **lower bound**
    against a threshold instead of trusting the single realized point estimate.
    Defaults to expectancy / profit_factor / sqn / win_rate; pass `metrics` to
    override. `trades` may be a TradeLog or a plain return array."""
    r = trades.r if isinstance(trades, TradeLog) else np.asarray(trades, dtype=float)
    r = r[~np.isnan(r)]
    metrics = metrics or _DEFAULT_METRICS
    if len(r) == 0:
        nan = float("nan")
        return {name: CI(nan, nan, nan, alpha) for name in metrics}

    rng = np.random.default_rng(seed)
    n = len(r)
    draws = {name: np.empty(n_boot) for name in metrics}
    for i in range(n_boot):
        s = rng.choice(r, size=n, replace=True)
        for name, fn in metrics.items():
            draws[name][i] = fn(s)

    out: "dict[str, CI]" = {}
    for name, fn in metrics.items():
        vals = draws[name]
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            # e.g. profit_factor on resamples that never have a loser -> all +inf
            lo = hi = float("inf")
        else:
            # inf draws (a resample with no losers) keep the percentile well
            # defined but trip a spurious inf-inf "invalid value" warning — mute it.
            with np.errstate(invalid="ignore"):
                lo = float(np.percentile(vals, alpha / 2 * 100))
                hi = float(np.percentile(vals, (1 - alpha / 2) * 100))
        out[name] = CI(point=float(fn(r)), low=lo, high=hi, alpha=alpha)
    return out


def p_value_positive(trades: TradeLog, metric: Metric = expectancy,
                     n_boot: int = 10_000, seed: int = 0) -> float:
    """One-sided bootstrap probability that the metric is > 0 (1.0 = strong)."""
    r = trades.r
    if len(r) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    draws = _resample(r, metric, n_boot, rng)
    finite = draws[np.isfinite(draws)]
    return float((finite > 0).mean()) if len(finite) else 0.0


def reality_check(trades: TradeLog, metric: Metric = expectancy,
                  metric_name: str = "expectancy", n_boot: int = 10_000,
                  alpha: float = 0.05, seed: int = 0) -> Verdict:
    """Point estimate + bootstrap CI + p-value, folded into a verdict:

      HELD     point > 0 and CI lower bound > 0
      FRAGILE  point > 0 but the CI straddles zero
      FAIL     otherwise

    This is the whole point of the package — the call backtesters never make."""
    ci = bootstrap_ci(trades, metric, n_boot, alpha, seed)
    p_pos = p_value_positive(trades, metric, n_boot, seed)
    if ci.point > 0 and ci.low > 0:
        label = "HELD"
    elif ci.point > 0:
        label = "FRAGILE"
    else:
        label = "FAIL"
    return Verdict(metric=metric_name, point=ci.point, ci=ci,
                   p_value=1 - p_pos, label=label)


def random_entry_null(prices, side: str, n_entries: int, hold: int, *,
                      tp: float = 2.0, sl: float = 1.0, n_sims: int = 1_000,
                      seed: int = 0) -> np.ndarray:
    """The strong test: expectancy of `n_sims` random-entry trade logs, each with
    `n_entries` random entries held `hold` bars under the same barriers, on the
    SAME prices. Compare your real edge against this distribution — did the signal
    beat coin-flip timing on this instrument? Returns the null expectancies."""
    from crucible.edge.simulator import barrier_trades, random_entries  # avoid cycle
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for i in range(n_sims):
        entries = random_entries(prices, n_entries, seed=int(rng.integers(1 << 31)))
        tl = barrier_trades(prices, entries, side=side, tp=tp, sl=sl, timeout=hold)
        out[i] = expectancy(tl.r) if len(tl) else np.nan
    return out


def detrended_timing_null(prices, holds, *, directions=None, scale=None,
                          n_samples: int = 2_000,
                          detrend: bool = True, seed: int = 42) -> np.ndarray:
    """Null per-trade-mean outcomes from random-*timing* trades matched to a
    strategy's holding periods and directions, on drift-removed price returns.

    Complements `random_entry_null` (which draws barrier-exit books on OHLC): this
    one resamples the asset's own bar-to-bar returns for the same holds/directions,
    detrended so the benchmark is asset-class-appropriate automatically — an equity
    index's structural long drift is removed the same way a currency's near-zero
    drift is, so no per-asset-class benchmark needs hand-picking. It isolates
    *timing* skill from riding the underlying drift.

    ``prices``: a price series (or 1-D array) covering the period. ``holds``:
    per-trade holding periods in bars. ``directions``: per-trade +1/-1 (all long
    when None).

    ``scale``: **per-trade multipliers that put the null in the same unit as your
    trade log's return column.** The raw draws are simple (fractional) returns; a
    log whose return column is a *simple return* (e.g. a rebalancing book) needs no
    scaling, so leave ``scale=None``. A log denominated in **R** (1R = entry->stop
    risk, `r = (exit-entry)/risk`) is a different unit: since
    `R = fractional_return * (entry / risk)` exactly, pass ``scale = entry / risk``
    per trade and the null comes back in R, matched to the observed statistic.
    Omitting it there is the classic units bug — comparing an expectancy in R against
    a null in fractional returns, off by the ~entry/risk factor (tens to hundreds),
    which lets almost any positive book clear the bar. Length must equal ``holds``.

    Returns an array of ``n_samples`` null per-trade-mean outcomes (in the scaled
    unit) — compare your strategy's mean per-trade return against its percentiles.
    Empty array if there aren't enough price bars.
    """
    px = np.asarray(getattr(prices, "values", prices), dtype=float)
    bar_returns = px[1:] / px[:-1] - 1.0
    bar_returns = bar_returns[np.isfinite(bar_returns)]
    if len(bar_returns) < 3:
        return np.array([])
    if detrend:
        bar_returns = bar_returns - bar_returns.mean()

    holds = np.nan_to_num(np.asarray(holds, dtype=float), nan=1.0)
    holds = np.clip(holds, 1, None).astype(int)
    directions = (np.ones(len(holds)) if directions is None
                  else np.nan_to_num(np.asarray(directions, dtype=float), nan=1.0))
    if scale is None:
        scale = np.ones(len(holds))
    else:
        scale = np.nan_to_num(np.asarray(scale, dtype=float), nan=1.0)
        if len(scale) != len(holds):
            raise ValueError(
                f"scale has length {len(scale)} but there are {len(holds)} trades; "
                "pass one entry/risk multiplier per trade (or None for a simple-return log)")

    rng = np.random.default_rng(seed)
    n_bars = len(bar_returns)
    null = np.empty(n_samples)
    for i in range(n_samples):
        sim = np.empty(len(holds))
        for j, (h, d, s) in enumerate(zip(holds, directions, scale)):
            h = min(int(h), n_bars)
            start = rng.integers(0, n_bars - h + 1)
            segment = bar_returns[start:start + h]
            sim[j] = s * d * (np.prod(1 + segment) - 1)
        null[i] = sim.mean()
    return null


# --------------------------------------------------------------------------- #
#  Block bootstrap — significance under SERIAL DEPENDENCE
#
#  The i.i.d. trade bootstrap and sign-permutation treat observations as
#  exchangeable, which breaks the time-clustering of a pooled multi-asset book
#  (correlated trades exit the same months). These operate on an ORDERED series
#  of PERIOD returns (e.g. monthly summed R on a gap-free calendar grid) and
#  resample contiguous blocks, so autocorrelation / cross-asset clustering is
#  preserved in the null.
# --------------------------------------------------------------------------- #

def _block_bootstrap_means(r: np.ndarray, block: int, n_boot: int, rng,
                           stationary: bool) -> np.ndarray:
    """Resample-mean distribution under a block bootstrap. `stationary=True` is the
    Politis–Romano stationary bootstrap (geometric block lengths, mean `block`);
    otherwise a circular fixed-length block bootstrap. Both vectorized over draws."""
    n = len(r)
    block = max(int(block), 1)
    if stationary:
        p = 1.0 / block
        idx = np.empty((n_boot, n), dtype=np.int64)
        restart = rng.random((n_boot, n)) < p
        jump = rng.integers(0, n, size=(n_boot, n))
        idx[:, 0] = jump[:, 0]
        for k in range(1, n):                       # sequential in time, vectorized over draws
            idx[:, k] = np.where(restart[:, k], jump[:, k], (idx[:, k - 1] + 1) % n)
    else:
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(n_boot, nb))
        offs = np.arange(block)
        idx = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(n_boot, nb * block)[:, :n]
    return r[idx].mean(axis=1)


def block_bootstrap_pvalue(returns, *, block: int = 6, n_boot: int = 10_000,
                           stationary: bool = False, seed: int = 0) -> float:
    """One-sided p-value that the mean of an ORDERED period-return series is > 0,
    preserving serial dependence via a block bootstrap.

    Feed an ordered series of PERIOD returns (e.g. monthly summed R over a gap-free
    calendar grid) — NOT a shuffled trade log. It zero-centers to impose H0
    (mean = 0), resamples contiguous blocks (circular, or `stationary=True` for the
    Politis–Romano stationary bootstrap), and returns P(bootstrap mean >= observed).
    Small p = the edge survives a correlation-preserving null. `block` is in periods:
    ~1 ≈ i.i.d.; larger blocks absorb more autocorrelation (widening the null when
    the series is positively autocorrelated — the clustering a trade-level test
    misses)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 1.0
    observed = float(r.mean())
    rng = np.random.default_rng(seed)
    boot = _block_bootstrap_means(r - observed, block, n_boot, rng, stationary)
    return float((np.sum(boot >= observed) + 1) / (n_boot + 1))


def block_bootstrap_ci(returns, *, block: int = 6, n_boot: int = 10_000,
                       stationary: bool = False, alpha: float = 0.05,
                       seed: int = 0) -> CI:
    """Percentile CI for the MEAN of an ordered period-return series under a block
    bootstrap. Wider than the i.i.d. :func:`bootstrap_ci` when the series is
    positively autocorrelated — the honest CI for a pooled, time-clustered book.
    Same block options as :func:`block_bootstrap_pvalue`."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return CI(float("nan"), float("nan"), float("nan"), alpha)
    rng = np.random.default_rng(seed)
    boot = _block_bootstrap_means(r, block, n_boot, rng, stationary)   # not centered → CI of the mean
    return CI(point=float(r.mean()),
              low=float(np.percentile(boot, alpha / 2 * 100)),
              high=float(np.percentile(boot, (1 - alpha / 2) * 100)), alpha=alpha)


# ── calendar-block CIs on a trade log ────────────────────────────────────────
# The i.i.d. bootstrap above treats every trade as independent. On a book pooled
# across markets, trades open in the same months overlap in calendar time and share
# the same shocks, so drawing them one at a time understates the noise. The
# resampler below draws whole calendar blocks of trades instead: every trade whose
# entry falls in a drawn block comes along together, so a shock common to that
# block moves the whole draw. Metrics are read off per-block sums, which keeps it
# exact for the decomposable metrics and vectorized over draws.

_BLOCK_METRICS = ("expectancy", "profit_factor", "win_rate")


def calendar_block_labels(entry_dates, block_months: int = 6) -> np.ndarray:
    """Integer calendar-block label per trade, from its entry date.

    Blocks are anchored on the calendar, not on the first trade: with
    ``block_months=6`` every trade lands in its half-year (Jan-Jun, Jul-Dec), with
    ``3`` in its quarter. Raises if a date is missing, since a trade that cannot be
    placed in time cannot be resampled with its neighbors."""
    import pandas as pd

    block_months = int(block_months)
    if block_months < 1:
        raise ValueError(f"block_months must be >= 1, got {block_months}")
    d = pd.to_datetime(pd.Series(entry_dates))
    if d.isna().any():
        raise ValueError(f"{int(d.isna().sum())} trade(s) have no entry_date; a calendar "
                         "block bootstrap needs every trade placed in time")
    months = d.dt.year.to_numpy() * 12 + (d.dt.month.to_numpy() - 1)
    return months // block_months


def calendar_block_metric_cis(trades: TradeLog, *, block_months: int = 6,
                              n_boot: int = 10_000, alpha: float = 0.05,
                              seed: int = 0) -> "dict[str, CI]":
    """Bootstrap CIs for expectancy, profit factor and win rate that resample whole
    CALENDAR BLOCKS of trades, the time-clustered companion to
    :func:`bootstrap_metric_cis`.

    Trades are grouped by the calendar block their ``entry_date`` falls in
    (:func:`calendar_block_labels`), and each draw picks that many blocks with
    replacement and pools their trades. Trades that overlap in calendar time and
    share a shock (a pooled multi-market book is the usual case) therefore move
    together, and the interval widens to match. When trades really are independent
    across blocks the two bootstraps agree in expectation; when they are not, the
    i.i.d. interval is the one that is too narrow.

    Point estimates are identical to the i.i.d. ones (same metric, same trades).
    Only blocks that contain a trade are drawn. With few blocks the percentile CI is
    coarse, so read ``n_blocks`` (:func:`calendar_block_labels`) alongside it.
    Requires an ``entry_date`` column; NaN returns are dropped with their dates."""
    if not isinstance(trades, TradeLog):
        raise TypeError("calendar_block_metric_cis needs a TradeLog with entry_date")
    dates = trades.col("entry_date")
    if dates is None:
        raise ValueError("calendar_block_metric_cis needs an entry_date column")
    r_all = trades.r
    keep = ~np.isnan(r_all)
    r = r_all[keep]
    nan = float("nan")
    if len(r) == 0:
        return {name: CI(nan, nan, nan, alpha) for name in _BLOCK_METRICS}

    labels = calendar_block_labels(np.asarray(dates)[keep], block_months)
    _, block = np.unique(labels, return_inverse=True)
    n_blocks = int(block.max()) + 1
    s_sum = np.bincount(block, weights=r, minlength=n_blocks)
    s_n = np.bincount(block, minlength=n_blocks).astype(float)
    s_gain = np.bincount(block, weights=np.where(r > 0, r, 0.0), minlength=n_blocks)
    s_loss = np.bincount(block, weights=np.where(r < 0, -r, 0.0), minlength=n_blocks)
    s_win = np.bincount(block, weights=(r > 0).astype(float), minlength=n_blocks)

    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    flat = (pick + n_blocks * np.arange(n_boot)[:, None]).ravel()
    w = np.bincount(flat, minlength=n_boot * n_blocks).reshape(n_boot, n_blocks).astype(float)

    n_draw = w @ s_n
    gp, gl = w @ s_gain, w @ s_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = {
            "expectancy": (w @ s_sum) / n_draw,
            "profit_factor": np.where(gl > 0, gp / gl, np.where(gp > 0, np.inf, 0.0)),
            "win_rate": (w @ s_win) / n_draw,
        }
    points = {"expectancy": expectancy(r), "profit_factor": profit_factor(r),
              "win_rate": win_rate(r)}
    out: "dict[str, CI]" = {}
    for name in _BLOCK_METRICS:
        vals = draws[name]
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            lo = hi = float("inf")
        else:
            with np.errstate(invalid="ignore"):
                lo = float(np.percentile(vals, alpha / 2 * 100))
                hi = float(np.percentile(vals, (1 - alpha / 2) * 100))
        out[name] = CI(point=float(points[name]), low=lo, high=hi, alpha=alpha)
    return out
