"""monitor: has a promoted edge decayed?

The gauntlet asks "is this edge real?" once, over a fixed log. This module asks the
question that follows a promotion: is it *still* real? It is a sequential detector over
a live `TradeLog`, measured against a baseline **frozen at promotion**.

Capital-free and stateless, like everything else here. It owns no clock and persists
nothing: a `TradeLog` and an `EdgeBaseline` in, a label out. The size decision it
implies (cut, halve, stop) is a capital-aware call and deliberately lives downstream.

Three commitments shape the API.

**1. Only a calibrated alarm can say DEGRADED.** Two detectors run. The CUSUM has a
stated false-alarm rate, so it alone drives `DEGRADED`. The rolling-window ratio has no
such calibration, so the worst it can say is `SLIPPING`. Running both is useful (one
fast and noisy, one slow and calibrated); letting the uncalibrated one govern the
decision is the mistake this split makes structurally impossible.

**2. No re-baselining.** A baseline recomputed from current data re-fits onto the
drifted reality, and the monitor can never fire. It looks entirely correct in review and
passes any test that does not span a real decay event. So `edge_monitor` takes an
`EdgeBaseline` and has no parameter that could rebuild one. Build it once, at promotion,
with `EdgeBaseline.from_log`, and freeze it.

**3. An undeflated baseline is allowed, but never silent.** Anchoring to the raw
in-sample expectancy (the number the parameters were optimized on) makes "half the
baseline" mean something other than half the edge you have. That is a legitimate choice
in a hurry, but it must be visible, so `deflated` rides in the verdict output rather
than being validated away. Same principle as `variant_count()` refusing a typed-in int.

**On the Gaussian assumption.** The ARL calibration below is Siegmund's approximation,
which assumes normal increments, and per-trade R is neither normal nor symmetric. That
sounds fatal and measurably is not: because the boundary sits 7 to 30 sigma away, the
CUSUM sums hundreds of increments before it can alarm, and the central limit theorem
does most of the work. Measured against resampled trade distributions, the nominal ARL0
holds to within a few percent for an ordinary 43%-win-rate shape, and drifts to about
+17% (conservative, meaning fewer false alarms than advertised) for a lottery-shaped
10%-win-rate book with +12R winners. Verify rather than assume: `empirical_arl` resamples
your own returns and reports the real number. Reach for it when your book is unusually
tail-heavy or when you design a fast boundary, since the approximation is weakest where
h is small.

**ARLs here are MEANS.** The run-length distribution is strongly right-skewed, so the
median is materially lower than the mean (often by a third). A published "median 474
trades to detect" and a mean of ~650 describe the same detector. `empirical_arl` returns
both; `CusumDesign.arl0` / `.arl1` are means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from crucible.edge.trade_log import TradeLog
from crucible.validation.thresholds import Thresholds

# Siegmund's continuity correction for the discrete-time boundary.
_SIEGMUND_SHIFT = 1.166

HOLDING = "HOLDING"
SLIPPING = "SLIPPING"
DEGRADED = "DEGRADED"


# ─────────────────────────────────────────────────────────────────────────────────────
# The frozen baseline
# ─────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EdgeBaseline:
    """What the validated edge promised, FROZEN at promotion. Denominated in R.

    In R rather than percent of account on purpose: change your risk-per-trade fraction
    and a percent-of-account series shifts for reasons that have nothing to do with the
    edge, which would show up here as spurious decay.
    """
    expectancy: float           # per-trade mean R. Deflated if you have it.
    sigma: float                # per-trade dispersion in R, sets the CUSUM's scale
    n_trades: int               # what the baseline was measured over
    trades_per_year: Optional[float] = None   # the opportunity set; None disables ch. 2
    n_variants: Optional[int] = None          # honest N this was corrected against
    deflated: bool = False      # False is legal, but it rides in the verdict

    def __post_init__(self) -> None:
        if not math.isfinite(self.expectancy) or self.expectancy <= 0:
            raise ValueError(
                f"baseline expectancy must be positive and finite, got {self.expectancy}. "
                "Monitoring the decay of an edge that was never positive is meaningless; "
                "the thing to run on a non-positive log is the gauntlet, not the monitor."
            )
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError(f"baseline sigma must be positive and finite, got {self.sigma}")
        if self.n_trades < 2:
            raise ValueError(f"baseline needs at least 2 trades, got {self.n_trades}")

    @classmethod
    def from_log(cls, trades: TradeLog, *, deflated_expectancy: Optional[float] = None,
                 n_variants: Optional[int] = None,
                 trades_per_year: Optional[float] = None) -> "EdgeBaseline":
        """Measure a baseline from the log the edge was validated on. Call this ONCE, at
        promotion, and freeze the result.

        `deflated_expectancy` is the search-corrected per-trade edge, if you have it.
        Supplying it sets `deflated=True`. Omitting it falls back to the log's sample
        mean, which is the number the parameters were optimized on and is therefore
        biased high; that is recorded as `deflated=False`, not silently accepted.
        """
        r = trades.r
        if trades_per_year is None:
            trades_per_year = _trades_per_year(trades)
        return cls(
            expectancy=float(deflated_expectancy if deflated_expectancy is not None
                             else r.mean()),
            sigma=float(r.std(ddof=1)),
            n_trades=int(r.size),
            trades_per_year=trades_per_year,
            n_variants=n_variants,
            deflated=deflated_expectancy is not None,
        )


def _trades_per_year(trades: TradeLog) -> Optional[float]:
    """Firing rate from entry dates, or None when the log carries no dates."""
    dates = trades.col("entry_date")
    if dates is None or len(dates) < 2:
        return None
    d = pd.to_datetime(pd.Series(dates)).sort_values()
    span_days = (d.iloc[-1] - d.iloc[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return None
    return float(len(d) * 365.25 / span_days)


# ─────────────────────────────────────────────────────────────────────────────────────
# CUSUM design: derive (k, h) from a stated shift and a stated false-alarm rate
# ─────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CusumDesign:
    """A one-sided lower CUSUM, designed rather than typed in.

    `k_r` is the midpoint reference value in R: (mu_0 + mu_1) / 2 for a target shifted
    mean mu_1. The statistic accumulates max(0, S + (k_r - x) / sigma) and alarms above
    `h_std`. Both parameterizations are reported because the raw one is what you sanity
    check by eye and the standardized one is what the ARL theory speaks.
    """
    k_r: float                  # midpoint reference value, in R
    k_std: float                # standardized slack, delta / 2
    h_std: float                # decision boundary, standardized
    h_r: float                  # the same boundary in R
    sigma: float
    detect_shift: float         # 0.5 = designed to detect a halving
    arl0: float                 # NOMINAL MEAN in-control run length, in trades
    arl1: float                 # NOMINAL MEAN trades to detect the design shift
    method: str = "siegmund-gaussian"

    def __str__(self) -> str:
        return (
            f"CUSUM design ({self.method})\n"
            f"  detects a shift to {self.detect_shift:.0%} of baseline\n"
            f"  k = {self.k_r:+.4f}R   h = {self.h_std:.2f} sigma ({self.h_r:.3f}R)\n"
            f"  nominal ARL0 = {self.arl0:,.0f} trades (mean, between false alarms)\n"
            f"  nominal ARL1 = {self.arl1:,.0f} trades (mean, to detect the design shift)\n"
            f"  Both are MEANS of a right-skewed distribution; the medians run well below.\n"
            f"  Gaussian approximation; check it with empirical_arl() on your own returns."
        )


def _siegmund_arl(xi: float, b: float) -> float:
    """Siegmund's approximation to the ARL of a one-sided CUSUM with unit-variance
    increments of mean `xi` and boundary `b`. Handles both signs of `xi`, so it serves
    the in-control (xi < 0) and out-of-control (xi > 0) cases from one expression."""
    y = 2.0 * xi * b
    if abs(y) < 1e-9:                      # the removable singularity at xi -> 0
        return b * b
    if -y > 700.0:                         # exp would overflow; the ARL is effectively unbounded
        return math.inf
    return (math.exp(-y) + y - 1.0) / (2.0 * xi * xi)


def _solve_h(k_std: float, target_arl0: float) -> float:
    """Smallest boundary whose nominal in-control ARL reaches `target_arl0`.
    ARL0 is monotone increasing in h, so plain bisection is exact enough."""
    lo, hi = 0.0, 1.0
    while _siegmund_arl(-k_std, hi + _SIEGMUND_SHIFT) < target_arl0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(
                f"cannot reach an in-control ARL of {target_arl0:,.0f} trades with a "
                f"standardized slack of {k_std:.5f}. The requested shift is too small "
                "relative to per-trade dispersion to detect at that false-alarm rate; "
                "raise detect_shift, or accept a shorter arl0_trades."
            )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _siegmund_arl(-k_std, mid + _SIEGMUND_SHIFT) < target_arl0:
            lo = mid
        else:
            hi = mid
    return hi


def cusum_design(baseline: EdgeBaseline, *,
                 thresholds: Thresholds = Thresholds()) -> CusumDesign:
    """Derive the CUSUM from what you want it to do, not from a typed-in threshold.

    You state the shift worth detecting (`monitor_detect_shift`, default a halving) and
    the false-alarm budget (`monitor_arl0_trades`); k and h follow, and the achieved
    detection latency comes back as `arl1` so it is a stated cost rather than something
    discovered three years into a live book.
    """
    shift = float(thresholds.monitor_detect_shift)
    if not 0.0 <= shift < 1.0:
        raise ValueError(f"monitor_detect_shift must be in [0, 1), got {shift}")

    mu0 = baseline.expectancy
    mu1 = shift * mu0
    delta = (mu0 - mu1) / baseline.sigma        # standardized shift magnitude
    k_std = delta / 2.0
    if k_std <= 0:
        raise ValueError("degenerate design: the target shift is zero in sigma units")

    h_std = _solve_h(k_std, float(thresholds.monitor_arl0_trades))
    b = h_std + _SIEGMUND_SHIFT
    return CusumDesign(
        k_r=0.5 * (mu0 + mu1),
        k_std=k_std,
        h_std=h_std,
        h_r=h_std * baseline.sigma,
        sigma=baseline.sigma,
        detect_shift=shift,
        arl0=_siegmund_arl(-k_std, b),
        arl1=_siegmund_arl(+k_std, b),
    )


# ─────────────────────────────────────────────────────────────────────────────────────
# Empirical calibration, because per-trade R is not Gaussian
# ─────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmpiricalARL:
    """Run length measured on YOUR return distribution rather than assumed Gaussian."""
    mean_run: float
    median_run: float           # the right-skew is large; report both or mislead
    nominal: float              # the design's matching Gaussian figure
    shift: float                # 1.0 = in control (ARL0); < 1 = detection latency
    n_sims: int
    censored_frac: float        # runs that hit max_run without alarming
    max_run: int
    seed: int

    @property
    def inflation(self) -> float:
        """Empirical mean / nominal mean. Above 1 means the Gaussian design is more
        conservative than advertised; below 1 means it fires sooner than you asked."""
        return self.mean_run / self.nominal if self.nominal else math.nan

    def __str__(self) -> str:
        which = "ARL0 (in control)" if self.shift == 1.0 else f"ARL1 (shift {self.shift:.0%})"
        note = "" if self.censored_frac < 0.05 else (
            f"\n  WARNING: {self.censored_frac:.0%} of runs censored at max_run={self.max_run}; "
            "the mean is a lower bound.")
        return (f"Empirical {which}: mean {self.mean_run:,.0f} / median "
                f"{self.median_run:,.0f} trades\n"
                f"  vs nominal mean {self.nominal:,.0f} ({self.inflation:.2f}x)"
                f"{note}\n  {self.n_sims} sims, seed {self.seed}")


def empirical_arl(design: CusumDesign, r_baseline: Sequence[float], *,
                  baseline_expectancy: float, shift: float = 1.0, n_sims: int = 1000,
                  max_run: Optional[int] = None, seed: int = 0) -> EmpiricalARL:
    """Measure the real run length by resampling the baseline's own returns.

    `shift` scales the true mean: 1.0 leaves the process in control and measures the
    false-alarm interval (ARL0); passing `design.detect_shift` measures how long the
    detector actually takes to notice the decay it was designed for (ARL1).

    The design's nominal figures assume Gaussian increments. Trade returns are skewed,
    so this bootstraps the actual distribution, recentered so its mean is exactly
    `shift * baseline_expectancy`, which is what "in control" (or "decayed") means here.
    The measured discrepancy is usually small, because a boundary many sigma away means
    the CUSUM aggregates enough increments for the CLT to apply, but it grows with tail
    weight and with a fast (small-h) design, which is exactly when you should check.

    Deterministic given `seed`.
    """
    r = np.asarray(r_baseline, dtype=float)
    if r.size < 2:
        raise ValueError("need at least 2 baseline returns to resample")
    if not 0.0 < shift <= 1.0:
        raise ValueError(f"shift must be in (0, 1], got {shift}")
    centered = r - r.mean() + float(baseline_expectancy) * float(shift)

    reference = design.arl0 if shift == 1.0 else design.arl1
    cap = int(max_run if max_run is not None
              else min(10 * reference, 200_000) if math.isfinite(reference)
              else 200_000)
    rng = np.random.default_rng(seed)

    run = np.full(n_sims, cap, dtype=np.int64)
    live = np.arange(n_sims)
    state = np.zeros(n_sims, dtype=float)
    steps, chunk = 0, 1024

    while steps < cap and live.size:
        m = min(chunk, cap - steps)
        draws = rng.choice(centered, size=(live.size, m), replace=True)
        inc = (design.k_r - draws) / design.sigma
        cur = state[live]
        hit = np.full(live.size, -1, dtype=np.int64)
        for j in range(m):
            cur = np.maximum(0.0, cur + inc[:, j])
            newly = (cur > design.h_std) & (hit < 0)
            hit[newly] = j + 1
        state[live] = cur
        fired = hit >= 0
        run[live[fired]] = steps + hit[fired]
        live = live[~fired]
        steps += m

    return EmpiricalARL(
        mean_run=float(run.mean()),
        median_run=float(np.median(run)),
        nominal=float(reference),
        shift=float(shift),
        n_sims=int(n_sims),
        censored_frac=float(live.size / n_sims),
        max_run=cap,
        seed=int(seed),
    )


# ─────────────────────────────────────────────────────────────────────────────────────
# The descriptive series
# ─────────────────────────────────────────────────────────────────────────────────────

def rolling_expectancy(trades: TradeLog, window: int) -> pd.Series:
    """Trailing mean R over a fixed number of TRADES (not calendar time).

    Descriptive, not a detector. Consecutive reads overlap heavily by construction: at
    138 trades a year a 200-trade window spans about 17 months, so two year-end readings
    share roughly a third of their trades and are nowhere near independent looks. Plot
    it, but do not count its threshold crossings as if they were.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2 trades, got {window}")
    s = pd.Series(trades.r, name=f"rolling_{window}_expectancy")
    return s.rolling(window).mean()


# ─────────────────────────────────────────────────────────────────────────────────────
# The verdict
# ─────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonitorVerdict:
    """Whether a promoted edge is still delivering what it promised."""
    label: str                          # HOLDING | SLIPPING | DEGRADED
    n_live: int

    # channel 1: the calibrated detector. The only one that can say DEGRADED.
    cusum_alarm: bool
    cusum_now: float                    # standardized, current level
    cusum_peak: float
    cusum_h: float
    alarm_index: Optional[int]          # 1-based live trade at which it first fired
    nominal_arl0: float

    # channel 2: the rolling ratio. Soft: the worst it can say is SLIPPING.
    baseline_expectancy: float
    recent_expectancy: Optional[float]  # None until the window fills
    edge_ratio: Optional[float]
    window: int

    # channel 3: the opportunity set. Soft, and skipped when the log has no dates.
    baseline_trades_per_year: Optional[float]
    live_trades_per_year: Optional[float]
    frequency_ratio: Optional[float]

    deflated: bool
    reasons: Tuple[str, ...]

    @property
    def pct_of_threshold(self) -> float:
        """Where the CUSUM sits as a fraction of its alarm boundary."""
        return self.cusum_now / self.cusum_h if self.cusum_h else math.nan

    def __str__(self) -> str:
        def _pct(v):
            return "n/a" if v is None else f"{v:.0%}"

        head = f"EDGE MONITOR: {self.label}   (n_live={self.n_live})"
        lines = [head, "-" * len(head)]
        peak_pct = self.cusum_peak / self.cusum_h if self.cusum_h else math.nan
        lines.append(
            f"  CUSUM      now {self.cusum_now:.2f} ({self.pct_of_threshold:.0%}) "
            f"peak {self.cusum_peak:.2f} ({peak_pct:.0%}) of h={self.cusum_h:.2f}"
            + (f"  ALARM at trade {self.alarm_index}" if self.cusum_alarm else "")
        )
        lines.append(
            f"  expectancy baseline {self.baseline_expectancy:+.4f}R  recent "
            + ("n/a (window not full)" if self.recent_expectancy is None
               else f"{self.recent_expectancy:+.4f}R  ratio {_pct(self.edge_ratio)}")
        )
        lines.append(f"  firing rate  ratio {_pct(self.frequency_ratio)}")
        if not self.deflated:
            lines.append("  ! baseline is NOT search-corrected; decay is measured from a "
                         "number that was biased high.")
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def edge_monitor(trades: TradeLog, baseline: EdgeBaseline, *,
                 design: Optional[CusumDesign] = None,
                 thresholds: Thresholds = Thresholds()) -> MonitorVerdict:
    """Has the live book's edge decayed from its frozen baseline?

    `trades` is the live log SINCE promotion, in the same R units the baseline was
    measured in. Deliberately has no parameter from which a baseline could be rebuilt:
    re-baselining is the bug this module exists to prevent.

    Only the CUSUM can return `DEGRADED`. The rolling ratio and the firing-rate ratio
    are uncalibrated, so they cap out at `SLIPPING`.
    """
    design = design or cusum_design(baseline, thresholds=thresholds)
    r = trades.r
    n = int(r.size)
    reasons: list[str] = []

    if n == 0:
        return MonitorVerdict(
            label=HOLDING, n_live=0, cusum_alarm=False, cusum_now=0.0, cusum_peak=0.0,
            cusum_h=design.h_std, alarm_index=None, nominal_arl0=design.arl0,
            baseline_expectancy=baseline.expectancy, recent_expectancy=None,
            edge_ratio=None, window=int(thresholds.monitor_window),
            baseline_trades_per_year=baseline.trades_per_year,
            live_trades_per_year=None, frequency_ratio=None, deflated=baseline.deflated,
            reasons=("no live trades yet; nothing to compare",),
        )

    # ── channel 1: the calibrated sequential detector ────────────────────────────────
    inc = (design.k_r - r) / design.sigma
    s, peak, alarm_index = 0.0, 0.0, None
    for i, step in enumerate(inc, start=1):
        s = max(0.0, s + float(step))
        peak = max(peak, s)
        if alarm_index is None and s > design.h_std:
            alarm_index = i
    cusum_alarm = alarm_index is not None
    if cusum_alarm:
        reasons.append(
            f"CUSUM crossed {design.h_std:.2f} sigma at live trade {alarm_index}; "
            f"designed to detect a fall to {design.detect_shift:.0%} of baseline with a "
            f"nominal false alarm every {design.arl0:,.0f} trades"
        )
    else:
        reasons.append(f"CUSUM at {peak / design.h_std:.0%} of threshold at its peak; "
                       "no calibrated alarm")

    # ── channel 2: the rolling ratio (soft) ──────────────────────────────────────────
    window = int(thresholds.monitor_window)
    recent = edge_ratio = None
    if n >= window:
        recent = float(r[-window:].mean())
        edge_ratio = recent / baseline.expectancy
        if edge_ratio < thresholds.monitor_slip_ratio:
            reasons.append(
                f"trailing {window}-trade expectancy is {edge_ratio:.0%} of baseline, "
                f"below the {thresholds.monitor_slip_ratio:.0%} soft line (uncalibrated: "
                "this rule has no stated false-alarm rate)"
            )
    else:
        reasons.append(f"only {n} live trades; the {window}-trade rolling read needs a "
                       "full window and is skipped")

    # ── channel 3: the opportunity set (soft) ────────────────────────────────────────
    live_tpy = _trades_per_year(trades)
    freq_ratio = None
    if baseline.trades_per_year and live_tpy:
        freq_ratio = live_tpy / baseline.trades_per_year
        if freq_ratio < thresholds.monitor_min_frequency_ratio:
            reasons.append(
                f"signal fires at {freq_ratio:.0%} of its baseline rate "
                f"({live_tpy:.0f}/yr vs {baseline.trades_per_year:.0f}/yr); annual R "
                "falls with the opportunity set even when per-trade expectancy holds"
            )
    elif baseline.trades_per_year is None:
        reasons.append("no baseline firing rate; the opportunity-set channel is off")
    else:
        reasons.append("live log carries no entry_date; the opportunity-set channel is off")

    if not baseline.deflated:
        reasons.append("baseline is NOT search-corrected (deflated=False), so the "
                       "reference point is the number the parameters were optimized on")

    # ── label: only the calibrated channel may escalate to DEGRADED ──────────────────
    soft_hit = (
        (edge_ratio is not None and edge_ratio < thresholds.monitor_slip_ratio)
        or (freq_ratio is not None and freq_ratio < thresholds.monitor_min_frequency_ratio)
    )
    label = DEGRADED if cusum_alarm else (SLIPPING if soft_hit else HOLDING)

    return MonitorVerdict(
        label=label, n_live=n, cusum_alarm=cusum_alarm, cusum_now=s, cusum_peak=peak,
        cusum_h=design.h_std, alarm_index=alarm_index, nominal_arl0=design.arl0,
        baseline_expectancy=baseline.expectancy, recent_expectancy=recent,
        edge_ratio=edge_ratio, window=window,
        baseline_trades_per_year=baseline.trades_per_year, live_trades_per_year=live_tpy,
        frequency_ratio=freq_ratio, deflated=baseline.deflated, reasons=tuple(reasons),
    )
