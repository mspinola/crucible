"""The honesty layer — the reason crucible.edge exists.

A positive expectancy on a trade log means nothing until you know it could not
have come from noise. These tools answer that with a confidence interval, a
resampling p-value, and — when you have the price series — a random-entry null.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from crucible.edge.trade_log import TradeLog
from crucible.edge.metrics import expectancy, profit_factor, sqn, win_rate

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


def detrended_timing_null(prices, holds, *, directions=None, n_samples: int = 2_000,
                          detrend: bool = True, seed: int = 42) -> np.ndarray:
    """Null per-trade-mean returns from random-*timing* trades matched to a
    strategy's holding periods and directions, on drift-removed price returns.

    Complements `random_entry_null` (which draws barrier-exit books on OHLC): this
    one resamples the asset's own bar-to-bar returns for the same holds/directions,
    detrended so the benchmark is asset-class-appropriate automatically — an equity
    index's structural long drift is removed the same way a currency's near-zero
    drift is, so no per-asset-class benchmark needs hand-picking. It isolates
    *timing* skill from riding the underlying drift.

    ``prices``: a price series (or 1-D array) covering the period. ``holds``:
    per-trade holding periods in bars. ``directions``: per-trade +1/-1 (all long
    when None). Returns an array of ``n_samples`` null per-trade-mean returns —
    compare your strategy's mean per-trade return against its percentiles. Empty
    array if there aren't enough price bars.
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

    rng = np.random.default_rng(seed)
    n_bars = len(bar_returns)
    null = np.empty(n_samples)
    for i in range(n_samples):
        sim = np.empty(len(holds))
        for j, (h, d) in enumerate(zip(holds, directions)):
            h = min(int(h), n_bars)
            start = rng.integers(0, n_bars - h + 1)
            segment = bar_returns[start:start + h]
            sim[j] = d * (np.prod(1 + segment) - 1)
        null[i] = sim.mean()
    return null
