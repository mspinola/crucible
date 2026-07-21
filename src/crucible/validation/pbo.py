"""Did the SELECTION overfit? — Probability of Backtest Overfitting + Deflated Sharpe.

`whites_reality_check` (permutation.py) asks whether the *best variant's edge* could
be noise. These two ask the complementary question: given that you searched N
configs and kept the best-in-sample one, **how much of that ranking survives out of
sample** — i.e. how badly did the *act of choosing* overfit?

  pbo_cscv         Combinatorially-Symmetric Cross-Validation (Bailey, Borwein,
                   López de Prado, Zhu 2017). Over every symmetric IS/OOS split of
                   the trial matrix, pick the best-in-sample config and measure how
                   often it lands below the OOS median. PBO = that fraction.
  deflated_sharpe  Bailey & López de Prado (2014). Deflate the winning Sharpe for
                   the number of trials AND the skew/kurtosis of its own returns —
                   the probability the true Sharpe clears the multiple-testing bar.

Both are pure: they consume a T x N performance matrix (rows = periods on a common
calendar, columns = the configs you searched) or plain return arrays. No capital,
no COT. Normal CDF/inverse-CDF come from stdlib `statistics.NormalDist`, so there is
no scipy dependency.

The honest read: PBO is a property of the SEARCH, not a verdict on any one book. A
book that already cleared a real forward holdout can still show a middling PBO — it
tells you how much to discount the *in-sample selection*, i.e. how many looks you
took. It is only as honest as the column set: pass EVERY config you tried, discards
included, or PBO reads optimistic.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Metric contract is COLUMNAR for speed: map a (rows x N) block of period returns
# to an N-vector of per-config scores, in one vectorized pass.
ColMetric = Callable[[np.ndarray], np.ndarray]
Returns = Union[Sequence[float], np.ndarray]

_NORM = NormalDist()
_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni, for the expected-max-Sharpe bound


def _sharpe_cols(block: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of every column: mean/std (ddof=1). Columns with zero or
    undefined dispersion score 0 (no information, not a divide-by-zero)."""
    if block.shape[0] < 2:
        return np.zeros(block.shape[1])
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)


def _as_matrix(M) -> np.ndarray:
    A = M.to_numpy(dtype=float) if isinstance(M, pd.DataFrame) else np.asarray(M, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"performance matrix must be 2-D (periods x configs), got shape {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(
            "performance matrix has NaN/inf — feed a DENSE grid (periods with no "
            "trade are 0.0 return, not missing). The matrix builder must fill gaps."
        )
    return A


@dataclass
class PBOResult:
    pbo: float                    # P(best-IS config lands below the OOS median) — headline
    logits: np.ndarray            # rank logit λ per split; pbo = mean(λ < 0)
    oos_below_zero: float         # P(chosen config's OOS performance < 0)
    degradation_slope: float      # OLS slope of chosen OOS vs IS score across splits;
                                  # negative under VARIED selection = performance decay.
                                  # (Trivially negative when one config dominates every
                                  # split — read it only alongside a spread of winners.)
    degradation_r2: float
    n_configs: int
    n_splits: int
    n_blocks: int

    @property
    def label(self) -> str:
        # Bailey et al. treat ~0.5 as "indistinguishable from a random selector".
        if self.pbo <= 0.10:
            return "ROBUST"
        if self.pbo <= 0.35:
            return "GUARDED"
        return "OVERFIT"

    def __str__(self) -> str:
        med = float(np.median(self.logits)) if len(self.logits) else float("nan")
        return "\n".join([
            "=" * 60,
            " PROBABILITY OF BACKTEST OVERFITTING  (CSCV)",
            "=" * 60,
            f"configs searched     : {self.n_configs}",
            f"blocks / splits      : S={self.n_blocks}  ->  {self.n_splits} symmetric splits",
            "-" * 60,
            f"PBO                  : {self.pbo:5.1%}     [{self.label}]",
            f"  P(chosen OOS < 0)  : {self.oos_below_zero:5.1%}",
            f"  median rank logit  : {med:+.2f}   (>0 = best-IS beats OOS median)",
            f"  OOS-vs-IS slope    : {self.degradation_slope:+.3f}  (R^2 {self.degradation_r2:.2f})",
            "=" * 60,
        ])


def pbo_cscv(M, S: int = 16, metric: ColMetric = _sharpe_cols) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV.

    `M` is a T x N matrix (DataFrame or array): rows are periods on a shared
    calendar, columns are the N configs you searched, cells are that config's
    return for that period. The rows are cut into `S` equal contiguous blocks
    (S even); for each of the C(S, S/2) ways to split the blocks into an IS half
    and its complementary OOS half we pick the config with the best IS `metric`,
    then read its RANK among all configs OOS. The relative rank maps to a logit
    λ; **PBO is the fraction of splits where the best-IS config falls below the
    OOS median** (λ < 0) — how often in-sample winning fails to carry over.

    Blocks stay contiguous so serial structure is preserved; if T isn't divisible
    by S the leading remainder rows are dropped (reported via n_blocks). `metric`
    is columnar: (rows x N) block -> N scores; the default is per-period Sharpe.

    Note PBO from ONE matrix is a noisy point estimate — for INDEPENDENT columns a
    single draw ranges widely around its ~0.5 noise center. Real strategy sweeps are
    far tamer because the configs are correlated variants of one book (a stable OOS
    ranking), but read PBO in bands (ROBUST / GUARDED / OVERFIT), not to the decimal.
    """
    A = _as_matrix(M)
    T, N = A.shape
    if S < 2 or S % 2 != 0:
        raise ValueError(f"S must be an even integer >= 2, got {S}")
    if S > T:
        raise ValueError(f"S={S} blocks but only {T} rows — need at least one row per block")
    if N < 2:
        raise ValueError(f"need >= 2 configs to rank, got {N}")

    block_len = T // S
    trimmed = A[T - block_len * S:]                      # drop the leading remainder
    row_groups = trimmed.reshape(S, block_len, N)        # (S blocks, rows-per-block, configs)

    logits, chosen_is, chosen_oos = [], [], []
    for combo in itertools.combinations(range(S), S // 2):
        is_mask = np.zeros(S, dtype=bool)
        is_mask[list(combo)] = True
        is_rows = row_groups[is_mask].reshape(-1, N)
        oos_rows = row_groups[~is_mask].reshape(-1, N)

        is_scores = metric(is_rows)
        oos_scores = metric(oos_rows)
        nstar = int(np.argmax(is_scores))               # the config you'd have deployed

        rank = int((oos_scores <= oos_scores[nstar]).sum())   # 1..N (ties count in)
        omega = rank / (N + 1)                          # in (0, 1): never 0 or 1
        logits.append(float(np.log(omega / (1.0 - omega))))
        chosen_is.append(float(is_scores[nstar]))
        chosen_oos.append(float(oos_scores[nstar]))

    logits = np.asarray(logits)
    is_arr, oos_arr = np.asarray(chosen_is), np.asarray(chosen_oos)
    slope, r2 = _ols(is_arr, oos_arr)

    return PBOResult(
        pbo=float((logits < 0).mean()),
        logits=logits,
        oos_below_zero=float((oos_arr < 0).mean()),
        degradation_slope=slope,
        degradation_r2=r2,
        n_configs=N,
        n_splits=len(logits),
        n_blocks=S,
    )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple:
    """Slope of y on x and its R^2 (0.0 slope / nan R^2 when x is degenerate)."""
    if len(x) < 2:
        return 0.0, float("nan")
    vx = x.var()
    if vx == 0:
        return 0.0, float("nan")
    slope = float(np.cov(x, y, ddof=0)[0, 1] / vx)
    denom = ((y - y.mean()) ** 2).sum()
    if denom == 0:
        return slope, float("nan")
    resid = y - (y.mean() + slope * (x - x.mean()))
    return slope, float(1.0 - (resid ** 2).sum() / denom)


# --------------------------------------------------------------------------- #
#  Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #

@dataclass
class DeflatedSharpe:
    observed_sharpe: float        # per-period Sharpe of the winning return series
    deflated_sharpe: float        # P(true SR > the multiple-testing threshold) in [0,1]
    sr0_threshold: float          # expected max per-period Sharpe under the null of N trials
    n_trials: int
    n_obs: int
    skew: float
    kurtosis: float               # non-excess (Pearson; 3.0 = normal)

    @property
    def label(self) -> str:
        if self.deflated_sharpe >= 0.95:
            return "SIGNIFICANT"
        if self.deflated_sharpe >= 0.90:
            return "MARGINAL"
        return "NOT SIGNIFICANT"

    def __str__(self) -> str:
        return "\n".join([
            "=" * 60,
            " DEFLATED SHARPE RATIO",
            "=" * 60,
            f"trials searched      : {self.n_trials}     observations: {self.n_obs}",
            f"observed Sharpe      : {self.observed_sharpe:+.3f}  (per period)",
            f"null max Sharpe SR0  : {self.sr0_threshold:+.3f}  (expected best of {self.n_trials} noise trials)",
            f"return skew / kurt   : {self.skew:+.2f} / {self.kurtosis:.2f}",
            "-" * 60,
            f"Deflated Sharpe      : {self.deflated_sharpe:5.1%}     [{self.label}]",
            "  (probability the true Sharpe clears the multiple-testing bar)",
            "=" * 60,
        ])


def _psr(sr: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star) given `n` observations and
    the return distribution's skew/kurtosis (Bailey & López de Prado 2012). The
    non-normality correction widens the SE when returns are left-skewed / fat-tailed."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or n < 2:
        return float("nan")
    z = (sr - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(_NORM.cdf(z))


def deflated_sharpe(trial_sharpes: Returns, *, returns: Returns,
                    n_trials: Optional[object] = None) -> DeflatedSharpe:
    """Deflated Sharpe Ratio for the winning config out of a search.

    `trial_sharpes` is the per-period Sharpe of every config you SCORED (its spread
    sets how high a Sharpe the search could hit by luck); `returns` is the winning
    config's own per-period return series (its Sharpe is the observed number, its
    skew/kurtosis correct the significance test for non-normality).

    The deflation threshold SR0 is the expected MAXIMUM Sharpe of N independent
    noise trials — grow the search and the bar rises. DSR = PSR evaluated at SR0:
    the probability the winner's true Sharpe actually clears that bar. Read ">= 95%"
    the way you'd read a passed significance test.

    **`n_trials` separates how many configs you TRIED from how many you SCORED.** By
    default N is the number of trial Sharpes supplied, which is right only when every
    config produced a scoreable result. It is wrong whenever a config errored or was
    too thin to score, and badly wrong when one search spans several sweeps — scanning
    45 markets and keeping the best is a search over 45 markets, not over the three
    configs of whichever one won. Pass the search's true size (an int, or a
    `SearchSpaceLog`, whose `session_n_variants` is read) and the bar rises to match.
    Supplying the scored count when more were tried silently undercounts the search
    and flatters the winner — the exact failure the ledger exists to prevent."""
    sr_trials = np.asarray(trial_sharpes, dtype=float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    scored = len(sr_trials)
    n = len(r)
    if scored < 2:
        raise ValueError(
            f"need >= 2 trial Sharpes to estimate the search's variance, got {scored}")
    if n < 2:
        raise ValueError(f"need >= 2 returns to compute the winning Sharpe, got {n}")

    # a SearchSpaceLog is accepted directly; duck-typed to avoid coupling the modules
    ledger_n = getattr(n_trials, "session_n_variants", None)
    N = scored if n_trials is None else int(ledger_n if ledger_n is not None else n_trials)
    if N < scored:
        raise ValueError(
            f"n_trials={N} is fewer than the {scored} trial Sharpes supplied; a search "
            "cannot have tried fewer configs than it scored")

    sd = r.std(ddof=1)
    sr_obs = float(r.mean() / sd) if sd > 0 else 0.0
    z = (r - r.mean()) / r.std(ddof=0) if r.std(ddof=0) > 0 else np.zeros(n)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())   # non-excess: 3.0 under normality

    var_sr = float(np.var(sr_trials, ddof=1))
    # Expected max of N standard-normal draws (Bailey/LdP): a two-point blend of the
    # (1 - 1/N) and (1 - 1/(N e)) quantiles, scaled by the trial-Sharpe dispersion.
    sr0 = np.sqrt(var_sr) * (
        (1.0 - _EULER_GAMMA) * _NORM.inv_cdf(1.0 - 1.0 / N)
        + _EULER_GAMMA * _NORM.inv_cdf(1.0 - 1.0 / (N * np.e))
    )
    dsr = _psr(sr_obs, float(sr0), n, skew, kurt)

    return DeflatedSharpe(
        observed_sharpe=sr_obs,
        deflated_sharpe=dsr,
        sr0_threshold=float(sr0),
        n_trials=N,
        n_obs=n,
        skew=skew,
        kurtosis=kurt,
    )
