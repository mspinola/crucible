"""Why the firing-rate channel is capped at SLIPPING, re-derivable without private data.

`docs/edge_monitor.md` records the measurements that killed the calibrated frequency
detector, taken on a real book that cannot be shipped. These pin the two general claims
those measurements rest on, synthetically, so the conclusion can be re-checked and so
nobody rebuilds the detector on the assumption it was never tried.

The claims:

  1. Arrivals that are OVERDISPERSED relative to Poisson break a Poisson-designed CUSUM
     in the dangerous direction: it fires far more often than its stated budget.
  2. Calibrating the boundary empirically fixes the bias but not the problem, because
     the calibration itself is only as good as the number of periods observed, and a
     low-frequency book does not supply enough of them.

Kept deliberately cheap. These are structural facts with wide margins, not tuning.
"""
import numpy as np
import pytest

LAM = 2.8                 # counts per month, ~34 trades/yr
DISPERSION = 1.3          # variance / mean, measured at 1.31 on the real book


def _overdispersed(rng, shape, lam=LAM, dispersion=DISPERSION):
    """Negative-binomial counts with the given mean and variance/mean ratio."""
    r = lam / (dispersion - 1.0)
    return rng.negative_binomial(r, r / (r + lam), shape).astype(float)


def _poisson(rng, shape, lam=LAM):
    return rng.poisson(lam, shape).astype(float)


def _k(lam=LAM, shift=0.5):
    """Poisson CUSUM reference for detecting a fall to `shift` x lam."""
    return (lam - shift * lam) / (np.log(lam) - np.log(shift * lam))


def _arl(sampler, h, *, seed, m=1500, n=1200):
    """Mean run length of the lower count-CUSUM, in periods."""
    rng = np.random.default_rng(seed)
    x = sampler(rng, (m, n))
    s = np.zeros(m)
    first = np.full(m, n + 1, dtype=float)
    alive = np.ones(m, dtype=bool)
    k = _k()
    for i in range(n):
        s = np.maximum(0.0, s + (k - x[:, i]))
        hit = alive & (s > h)
        first[hit] = i + 1
        alive &= ~hit
        if not alive.any():
            break
    return float(first.mean())


def _solve_h(sampler, target, *, seed, iters=12):
    lo, hi = 0.05, 40.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _arl(sampler, mid, seed=seed) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_overdispersion_breaks_a_poisson_design_in_the_dangerous_direction():
    """Claim 1. The design method is sound on data matching its model, and the model is
    what the book violates. Measured on the real book at 0.32x; the synthetic analogue
    only has to reproduce the direction and rough size."""
    target = 120.0                                   # periods
    h = _solve_h(_poisson, target, seed=1)

    nominal = _arl(_poisson, h, seed=2)
    real = _arl(_overdispersed, h, seed=2)

    assert nominal == pytest.approx(target, rel=0.35), "the design should hit its own model"
    assert real < nominal * 0.75, (
        f"overdispersed arrivals should fire MORE often than stated, got {real:.0f} "
        f"vs {nominal:.0f}")


def test_calibrating_the_boundary_removes_the_bias():
    """Solving h against the true process is the honest fallback, and it works. It is
    the uncertainty in that solution, not its centre, that kills the detector."""
    target = 120.0
    h_poisson = _solve_h(_poisson, target, seed=1)
    h_real = _solve_h(_overdispersed, target, seed=1)

    assert h_real > h_poisson, "an overdispersed process needs a WIDER boundary"
    assert _arl(_overdispersed, h_real, seed=5) == pytest.approx(target, rel=0.4)


def test_the_calibration_is_only_as_good_as_the_periods_observed():
    """Claim 2, and the one that decides it. Calibrate h from a SAMPLE of periods rather
    than the true process, as any real book must, and the delivered budget scatters. The
    scatter shrinks with the number of periods, which is why this is a property of the
    book's firing rate rather than of the statistics.

    A book at ~34 trades/yr supplies about 82 monthly periods in a 7-year window. On the
    real book that left the delivered budget spanning 5.3 to 77.9 years, a 14.6x range.
    """
    target = 120.0
    truth = _arl(_overdispersed, _solve_h(_overdispersed, target, seed=1), seed=9)
    rng = np.random.default_rng(0)

    spreads = {}
    for n_periods in (80, 2000):
        delivered = []
        for rep in range(4):
            sample = _overdispersed(rng, (n_periods,))

            def resample(r, shape, s=sample):
                return r.choice(s, shape, replace=True)

            h = _solve_h(resample, target, seed=30 + rep)
            delivered.append(_arl(_overdispersed, h, seed=40 + rep))
        spreads[n_periods] = max(delivered) / max(min(delivered), 1e-9)

    assert spreads[80] > 2.0, (
        f"80 periods should not pin the budget down; got a {spreads[80]:.1f}x spread")
    assert spreads[2000] < spreads[80], (
        "more periods must calibrate better, or the limit is not about sample size: "
        f"{spreads[2000]:.1f}x at 2000 vs {spreads[80]:.1f}x at 80")
    assert truth > 0


def test_the_channel_is_still_capped_at_slipping():
    """The conclusion, asserted on the code rather than left to the docs. If a future
    change lets the frequency ratio reach DEGRADED, this fails and sends the reader to
    the measurements that say why it must not."""
    import inspect

    from crucible.validation import monitor

    src = inspect.getsource(monitor.edge_monitor)
    assert "soft_hit" in src, "the soft-channel gate was renamed or removed"
    # DEGRADED is reachable only through the calibrated CUSUM's alarm
    label_line = [ln for ln in src.splitlines() if "DEGRADED if" in ln]
    assert label_line, "the label rule changed shape; re-read docs/edge_monitor.md"
    assert "cusum_alarm" in label_line[0], (
        "only the calibrated detector may escalate to DEGRADED. See "
        "'Why the firing-rate channel stays uncalibrated' in docs/edge_monitor.md: "
        "three arrival-process detectors were built and all three delivered a "
        "false-alarm rate 2-3x worse than stated.")
