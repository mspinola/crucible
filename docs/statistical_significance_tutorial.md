# From Trade Log to Verdict: The Statistics of a Significant Edge

**A tutorial on the statistical techniques in `crucible` (and, to a lesser extent,
`pardo_quant_framework`), and how each one contributes to declaring a trade log
statistically significant — with references to the source literature.**

---

## 0. What "statistically significant trade log" actually means

A backtester hands you a rising equity curve and a headline number — "profit factor
1.37, 214 trades." The number is *real* in the sense that it happened on the data. The
question this tutorial is about is different:

> Given how few trades I have, and how many variants I tried before landing on this
> one, could a curve this good have come from **no edge at all**?

Answering that turns a point estimate into a **verdict**. Two distinct failure modes
have to be ruled out (the framing is Aronson's, carried into
`pardo_quant_framework/docs/edge_validation_framework.md`):

1. **Luck mistaken for skill** — a small sample, or a large search, threw up a good
   number by chance. *(Statistical validity.)*
2. **A real pattern that doesn't travel** — genuine in-sample, gone out-of-sample or on
   another market. *(Robustness / generalization.)*

`crucible` is the tool that answers these at the **trade-log level**, before any
capital, position sizing, or equity curve enters the picture. `pardo_quant_framework`
wraps the same primitives in a staged, gated pipeline for a real COT-based book.

Everything below is organized as a pipeline: **describe the edge → quantify sampling
noise → rule out data-mining luck → rule out drift → confirm out-of-sample → account
for correlation**. Each section names the technique, the code that implements it, the
statistical logic, and where to read the primary source.

---

## 1. The substrate: a risk-normalized trade log (R-multiples)

**Code:** [`edge/trade_log.py`](../src/crucible/edge/trade_log.py),
[`edge/simulator.py`](../src/crucible/edge/simulator.py)

Every technique downstream operates on one object: a `TradeLog` whose required column
`r` is the per-trade return in **R-multiples** — profit measured in units of the risk
taken at entry — the entry-to-stop distance (`1R = entry − stop`, which the barrier
simulator sizes as `sl × ATR`). R-normalization is what lets
returns from different instruments and volatility regimes pool into one sample that the
statistics can treat as draws from a single distribution.

The `TradeLog` is deliberately agnostic about *how* the trades were produced. A
hand-coded moving-average rule, a set of discretionary fills exported from a broker or a
RealTest run, and an ML take/skip filter all reduce to the same schema — a column of
R-multiples (plus optional MFE/MAE, holding period, entry/exit dates). The honesty layer
never sees the strategy; it sees only the returns, which is exactly why one set of tests
serves rule-based and model-based books alike.

When you *do* generate the log from an entry rule, the barrier simulator that manufactures
it is deliberately **look-ahead-free**: barriers are sized off the signal bar — the bar
the rule fires on, known at entry — and exits are scanned forward from the entry bar
(`simulator.py:64`, `:75`). This matters because every p-value later assumes each `r` was
knowable only at the moment of the trade — a single peek into the future contaminates the
whole null distribution.

> **Sources.** The R-multiple as the unit of trade evaluation: Van Tharp, *Trade Your
> Way to Financial Freedom* (origin of R and SQN, below). Risk-normalized, volatility-
> scaled position/return accounting: Carver, *Systematic Trading*, **Ch. 9 "Volatility
> Targeting"** and **Ch. 10 "Position Sizing"**. The leakage-free barrier construction is
> the same geometry ML uses to label forward outcomes (López de Prado, *Advances in
> Financial Machine Learning*, **§3.4 "The Triple-Barrier Method"**, cross-referenced in
> §7) — but here it turns *any* entry rule into trades, ML or not.

---

## 2. Describing the edge: capital-free metrics

**Code:** [`edge/metrics.py`](../src/crucible/edge/metrics.py)

Before any significance test, you summarize the sample. These are point estimates — they
*describe*, they do not yet *defend*.

| Metric | Definition (code) | Reads on |
|---|---|---|
| **Expectancy** | `wr·avg_win − lr·avg_loss` (in R), `metrics.py:28` | mean profit per trade |
| **Profit factor** | gross win / gross loss, `:41` | reward-to-risk of the whole book |
| **Payoff ratio** (a.k.a. RR / risk-reward) | avg win / avg loss, `:51` | terminal reward-to-risk geometry |
| **SQN** | `mean/std · √min(n,100)`, `:60` | *signal-to-noise* — the risk-adjusted quality score |
| **Excursion / E-ratio** | mean MFE / mean\|MAE\|, `:72`,`:81` | is there directional edge *before* the exit rule? |
| **Time asymmetry** | avg bars in wins / avg bars in losses, `:87` | "let winners run, cut losers" |
| **Exit efficiency** | captured / available MFE, `:98` | how much of the move the exit banked |

The one to single out is **SQN** (System Quality Number, Van Tharp):
`mean(R) / std(R) × √n`. It is a Sharpe-like *t*-statistic on the per-trade returns — the
same quantity a significance test formalizes. A high mean means nothing if the standard
deviation is huge or `n` is tiny; SQN is the first hint of whether the sample can support
a claim at all.

> **Sources.**
> - Expectancy, profit factor, payoff, drawdown and the rest of the classic evaluation
>   battery: **Pardo, *The Evaluation and Optimization of Trading Strategies*, 2nd ed.
>   (2008), the "Evaluation of Trading Strategies" chapter** — this is the canonical list
>   `edge_report` reproduces.
> - **SQN**: Van Tharp, *Trade Your Way to Financial Freedom*, 2nd ed. (2007), ch. on
>   "The System Quality Number." Both codebases use Van Tharp's `√min(n,100)` cap:
>   crucible's `sqn()` (`metrics.py:60`) is the single source of truth, and the
>   framework's `PardoSQNEvaluator.calculate_sqn` delegates to it. (An earlier Pardo
>   variant used `√100` always to normalize low-frequency books across asset classes,
>   but that credited small samples with confidence they hadn't earned — inflating a
>   30-trade SQN by √(100/30) ≈ 1.8× — and was replaced. Its walk-forward ratio
>   WFE_SQN stays computed on the sample-size-independent per-trade quality (mean/std),
>   so a fold's robustness score isn't distorted by IS/OOS window-length asymmetry.)
> - Excursion (MFE/MAE) analysis: Sweeney, *Campaign Trading*; Curtis Faith, *Way of the
>   Turtle* (the E-ratio) — cited in the `metrics.py` module docstring.
> - The *risk-adjusted* framing (why std matters as much as mean): Carver, *Advanced
>   Futures Trading Strategies* (2021), the chapters on the **Sharpe ratio and evaluating
>   returns** — Carver stresses that a Sharpe/SQN is itself an estimate with a wide error
>   bar, which is exactly what §3 addresses.

---

## 3. Quantifying sampling noise: the bootstrap confidence interval

**Code:** [`edge/stats.py`](../src/crucible/edge/stats.py) — `bootstrap_ci`,
`p_value_positive`, and `bootstrap_metric_cis` (the whole metric set in one resample pass).
pqf's [`validation/bootstrap.py`](../../pardo_quant_framework/src/validation/bootstrap.py)
now delegates here.

A point estimate of expectancy on 60–200 trades badly understates how much it could
have wobbled. The **bootstrap** turns the single number into a distribution:

```
for i in 1..10000:           # stats.py:54  _resample
    draw = sample r WITH REPLACEMENT, size n
    record metric(draw)
CI = [2.5th percentile, 97.5th percentile]     # stats.py:76
```

Resampling the trade log with replacement simulates "other histories you could plausibly
have drawn from the same edge." The **2.5–97.5 percentile band** is the 95% confidence
interval; `p_value_positive` reports the fraction of resamples where the metric stayed
positive.

Why it is the honest read: it makes no normality assumption (trade returns are skewed and
fat-tailed), and it works for *any* metric — expectancy, profit factor, SQN — not just the
mean. On small samples the band is wide, and that width **is the message**.

crucible's gauntlet gates on this: its **STRONG** gate requires the **CI lower bound**, not
the point estimate, to clear each threshold (`gate_strong`, §11). This directly fixes the
"PF 1.37 on 60 trades treated as a clean pass" failure — the same bar pqf's Stage 4 now
enforces through crucible.

> **Sources.**
> - **Aronson, *Evidence-Based Technical Analysis* (EBTA, 2006), Ch. 4 "Statistical
>   Analysis" (p. 165) and Ch. 5 "Hypothesis Tests and Confidence Intervals" (p. 217)** —
>   sampling distributions, the bootstrap, and confidence intervals for trading
>   statistics. This is the direct antecedent of `bootstrap_ci`.
> - AFML **Ch. 14 "Backtest Statistics"** — the family of statistics a backtest should
>   report as distributions, not points.
> - Carver, *Systematic Trading*, **Appendix C "Portfolio Optimisation → More details on
>   bootstrapping"** — bootstrap resampling applied to strategy/portfolio estimates.

---

## 4. The verdict: folding point + CI + p-value into a label

**Code:** [`edge/stats.py`](../src/crucible/edge/stats.py) — `reality_check`, `Verdict`
(`stats.py:93`)

`reality_check` is the call the README calls "the whole point of the package." It collapses
the three numbers into a decision:

```
HELD     point > 0  AND  CI lower bound > 0     # the edge clears zero across resamples
FRAGILE  point > 0  BUT  CI straddles zero       # positive, but indistinguishable from noise
FAIL     otherwise
```

`FRAGILE` is the state a backtester never shows you: a positive expectancy whose confidence
interval includes zero. The equity curve looked like an edge; the statistics say *don't
size it up.* This is significance testing expressed as an operating instruction rather than
a p-value to be argued over.

> **Sources.** The philosophy — a rule earns belief only when the evidence clears a
> pre-set statistical bar, not when it merely looks good — is the thesis of **EBTA Ch. 3
> "The Scientific Method and Technical Analysis" (p. 103)** and **Ch. 5 (p. 217)**. The
> "hard gate, no discretionary override" posture is spelled out in
> `edge_validation_framework.md` → *Two-Tier Gating Philosophy*.

---

## 5. Ruling out data-mining luck: permutation tests

This is the heart of the significance story and the reason Aronson & Masters matter.

### 5a. Sign-permutation test (one strategy)

**Code:** [`validation/permutation.py`](../src/crucible/validation/permutation.py) —
`sign_permutation_pvalue`

```
observed = mean(r)
for k in 1..5000:                              # permutation.py:42
    flip each trade's sign at random (±1)
    record mean(signs · |r|)
p = (# permuted means ≥ observed + 1) / (N + 1)
```

The **null hypothesis** is "no directional skill" — under it, each trade's *sign* is a coin
flip while its *magnitude* is whatever it was. Shuffling signs builds the distribution of
outcomes a skill-less system would produce on these same magnitudes; the p-value is how
often chance matches or beats you. This is **Timothy Masters' Monte Carlo Permutation
Method**, the public-domain alternative to White's patented Reality Check.

### 5b. Šidák correction (you tried N variants)

**Code:** `permutation.py:47` — `sidak_correction`; `corrected = 1 − (1 − p)^N`

If you quietly tried 50 parameter sets and reported the best, its raw p-value is a lie of
selection. Šidák asks: *what's the chance the best of N independent searches looks this
good by luck?* It is the conservative fallback when you only know the **count** of variants.
crucible's **REAL** gate applies it in the gauntlet (pass your `n_variants`); pqf's Stage 3
feeds it the same count from its search-space log.

### 5c. White's Reality Check (you have every variant's returns)

**Code:** `permutation.py:59` — `whites_reality_check`

```
for each permutation:                          # permutation.py:84
    flip signs for EVERY variant
    record the BEST mean across all variants   # the max-statistic
compare observed best against this "distribution of the best"
```

Taking the **maximum inside each permutation** is what corrects for the size of the search:
you compare your winner not against zero, but against *the best number a pure-noise search
of the same size would have thrown up.* The docstring's warning is load-bearing — you must
pass **every variant including the discards**, or the correction is toothless.

> **Sources.**
> - **EBTA Ch. 6 "Data-Mining Bias: The Fool's Gold of Objective TA" (p. 255)** — the
>   definitive treatment of data-mining bias, the Monte Carlo permutation method, and
>   White's Reality Check, applied in the **Ch. 8 case study (p. 389)** across 6,402 rules.
> - EBTA **Acknowledgments (p. ix)** credits **Timothy Masters** with innovating the Monte
>   Carlo permutation method and placing it in the public domain — the exact method in
>   `sign_permutation_pvalue`.
> - **Aronson & Masters, *Statistically Sound Machine Learning for Algorithmic Trading*
>   (SSML, 2013)** — the applied companion; see the Introduction's **"Performance Criteria"**
>   and the permutation-test / selection-bias sections (the book is organized by topic
>   around the TSSB tool).
> - White, H. (2000), "A Reality Check for Data Snooping," *Econometrica* 68(5) — the
>   original max-statistic bootstrap `whites_reality_check` reimplements.
> - Multiple-comparisons / overfitting the search itself: AFML **Ch. 11 "The Dangers of
>   Backtesting"** and **Ch. 12 "Backtesting through Cross-Validation."**

### Did the *selection* overfit? — PBO & deflated Sharpe

**Code:** [`validation/pbo.py`](../src/crucible/validation/pbo.py) — `pbo_cscv`, `deflated_sharpe`

White's Reality Check asks whether the best variant's *edge* is noise. Two companion tools ask
the complementary question: given that you searched N configs and kept the best-in-sample one,
**how much did the act of choosing overfit?**

- **PBO — Probability of Backtest Overfitting** (`pbo_cscv`) via Combinatorially-Symmetric
  Cross-Validation. Feed a `T×N` performance matrix (periods × the configs you searched). Over
  every symmetric IS/OOS split of the period blocks it picks the best-in-sample config and reads
  its **rank out-of-sample**; PBO is the fraction of splits where the in-sample winner lands
  **below the OOS median**. Read it in bands — `ROBUST` (≤0.10) / `GUARDED` / `OVERFIT` — not to
  the decimal, and (like White's) pass *every* config you tried or it reads optimistic.

- **Deflated Sharpe Ratio** (`deflated_sharpe`). The winner's Sharpe must clear a bar that
  **rises with the number of trials**: `SR0` is the expected maximum Sharpe of N noise trials,
  and the DSR is the probability the winner's *true* Sharpe beats it — corrected for the return
  series' own **skew and kurtosis** (fat left tails widen the error bar). Read `≥ 95%` like a
  passed significance test.

Where the permutation test corrects the *p-value* for the search, these correct the *IS ranking*
and the *Sharpe* for it — the same multiple-testing disease, caught two more ways. Capital-free
(stdlib `NormalDist`, no scipy).

> **Sources.** **PBO / CSCV**: Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of
> Backtest Overfitting," *Journal of Computational Finance*; **AFML Ch. 11–12**. **Deflated /
> Probabilistic Sharpe**: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio," *Journal of
> Portfolio Management*, and (2012) the Probabilistic Sharpe Ratio; **AFML Ch. 14 "Backtest
> Statistics."**

---

## 6. Ruling out drift: the random-entry / detrended benchmark

**Code:** `edge/stats.py:115` — `random_entry_null` (crucible);
[`pardo .../validation/benchmark.py`](../../pardo_quant_framework/src/validation/benchmark.py)
— `detrended_benchmark_test` (framework)

Beating zero is not enough on an instrument that drifts up. The right null is *"did my
signal beat coin-flip timing on this same instrument?"*

- **crucible** `random_entry_null`: run `n_sims` trade logs with **random entries**, same
  barriers, same prices; compare your real expectancy against that distribution.
- **framework** `detrended_benchmark_test`: build randomly-timed trades **matched to your
  actual directions and holding periods**, on a **drift-removed** return series, and require
  your per-trade expectancy to beat the 95th percentile of that no-skill distribution
  (`benchmark.py:63-73`).

Detrending is what isolates *timing skill* from *riding the market*. It also makes the
benchmark automatically asset-class-appropriate: an equity index's structural long drift is
removed the same way a currency's near-zero drift is, so no hand-picked per-class benchmark
is needed.

> **Sources.**
> - **EBTA Appendix "Proof That Detrending Is Equivalent to Benchmarking Based on Position
>   Bias" (p. 475)** — the theoretical justification for the detrended benchmark; also
>   **Ch. 1 "Objective Rules and Their Evaluation" (p. 15)** on benchmarking a rule against
>   its position bias.
> - SSML Introduction, **"Model Performance Versus Financial Performance"** and **"Financial
>   Relevance and Generalizability"** — beating a naive-exposure benchmark, not just zero.

---

## 7. The ML track: honest labels, and is the signal real?

**Code:** [`pardo .../ml/labels.py`](../../pardo_quant_framework/src/ml/labels.py) —
`compute_triple_barrier_labels`; barrier version in
[`crucible .../simulator.py`](../src/crucible/edge/simulator.py)

Before you can test a trade log you must define what a "trade outcome" *is*, mechanically
and without hindsight. The **triple-barrier method** labels each entry by whichever of three
events fires first: a profit barrier (`+tp·ATR`), a stop barrier (`−sl·ATR`), or a vertical
time barrier (`horizon`). Volatility-scaled (ATR) barriers keep the labels comparable across
regimes. Because the label *looks forward* until a barrier is touched, it creates the
leakage risk that §8's purge/embargo exists to neutralize.

> **Sources.** **AFML Ch. 3 "Labeling," §3.4 "The Triple-Barrier Method"** (and §3.2–3.3 on
> fixed-time-horizon and dynamic thresholds) — the exact construction `labels.py` implements.
> Pairs with **§3.5 "Learning Side and Size."**

### Meta-labeling (framework)

**Code:** [`pardo .../ml/meta_eval.py`](../../pardo_quant_framework/src/ml/meta_eval.py)

The ML layer does not predict direction; it predicts **whether to take or skip** a trade the
rules book already generated (meta-label = "did this trade win?"). This is López de Prado's
**meta-labeling**: keep a transparent primary model for *side*, add an ML filter for
*size/precision*, and judge it only if take/skip beats take-all out-of-sample.

> **Sources.** **AFML §3.6 "Meta-Labeling"** and **§3.7 "How to Use Meta-Labeling."**

### Is the ML score real? — `crucible.ml`

**Code:** [`ml/ic.py`](../src/crucible/ml/ic.py), [`ml/decay.py`](../src/crucible/ml/decay.py),
[`ml/redundancy.py`](../src/crucible/ml/redundancy.py), [`ml/pit.py`](../src/crucible/ml/pit.py)

Once a model emits a **score**, the same honesty question §4 asks of a trade log applies to the
score: does a higher score actually rank better outcomes, or is it noise, leakage, or a feature
wearing a new name? `crucible.ml` answers it capital-free (numpy/pandas only), on the model's
predictions rather than an equity curve.

- **Information Coefficient** (`information_coefficient`) — the Spearman **rank** correlation
  between a score and its realized label. Rank-based, so it's invariant to the label encoding
  (+1/−1 or 0/1) and to any monotonic transform of the score: it measures only whether higher
  scores line up with better outcomes. `alpha_gate(ic, min_ic=…)` raises below the bar — a
  PASS/FAIL you wire into a training loop to kill an edge-less or leaking model before it reaches
  a backtester. Computed **out-of-fold** (`fold_ic`), and — echoing §5 — its **sign-stability
  across folds** matters more than its magnitude: a weak-but-consistently-positive IC is a better
  sign than a strong one that flips.

- **Quantile decay** (`quantile_decay`) — bucket the score into equal-count quantiles and read
  the realized win rate per bucket. A genuine, well-ordered edge makes win rate climb
  **monotonically** Q1→Q5 (`.monotonic`, `.spread`); a flat or ragged profile is the tell of a
  score that ranks nothing. `decay_tearsheet` renders it as self-contained HTML.

- **Feature redundancy** (`redundancy_droplist`) — clusters features by |Spearman| / Cramér's V
  and keeps the highest-|IC| member of each cluster. This is the feature-space analogue of §10's
  N_eff: three features that are one feature in disguise are one bet, not three, and counting them
  as independent inflates any significance claim downstream.

- **Point-in-time slices** (`asof_window` / `window_before`) — a leakage-safe window so a live
  feature is built identically to its training twin: the feature-space cousin of §8's
  purge/embargo. §8 stops the *label* from peeking ahead; this stops the *features* from doing so.

> **Sources.** The Information Coefficient and the link between per-bet skill and portfolio
> performance: **Grinold & Kahn, *Active Portfolio Management*** (the IC and the Fundamental Law of
> Active Management) — outside the six-book set, but the canonical IC reference. Out-of-fold
> feature importance, judging features on unseen data: **AFML Ch. 8 "Feature Importance."** No-look-
> ahead feature construction: **AFML Ch. 7 §7.4** — the purge/embargo principle applied to features.
> Quantile-decay monotonicity is the standard factor-research check (the alphalens lineage).

---

## 8. Confirming out-of-sample: holdout, purge & embargo

**Code:** [`validation/holdout.py`](../src/crucible/validation/holdout.py) — `holdout`,
`split_train_test`

For a **fixed** strategy, the honest test is temporal: measure the edge early, freeze it,
confirm on a late period the analysis never touched. Two leakage controls make the split
real (`holdout.py:36-48`):

- **Purge:** a training trade must have both *entered and exited* before the split — a trade
  whose forward window straddles the boundary can't leak future information into the fitted
  side.
- **Embargo:** drop the first `embargo_weeks` of the test period, killing residual
  autocorrelation across the seam.

The `HoldoutResult` runs a full `reality_check` on each side and declares **the untouched
TEST period the verdict** (`holdout.py:56`, `:69`). Train is expected to look good — that's
where an edge would have been chosen; only test counts.

> **Sources.**
> - **AFML Ch. 7 "Cross-Validation in Finance," §7.4 "A Solution: Purged K-Fold CV"** — the
>   origin of purge + embargo for overlapping financial labels. Directly cited in the
>   framework doc's Stage 1.
> - **Pardo (2008), the "Walk-Forward Analysis" chapter** — the in-sample/out-of-sample
>   discipline this generalizes.
> - Carver, *Systematic Trading*, **Ch. 3 "Fitting"** — why in-sample results prove nothing
>   and the case for simple, robust, hard-to-overfit parameters.

---

## 9. Confirming it *keeps* working: walk-forward analysis & efficiency

**Code:** [`validation/walk_forward.py`](../src/crucible/validation/walk_forward.py) —
`walk_forward`, `_wfe`

One holdout is a single split; **walk-forward** rolls it through history: optimize params on
an in-sample window, apply the winner to the next untouched out-of-sample window, step
forward, repeat, then **stitch all OOS slices into one trade log**. If the stitched OOS edge
survives `reality_check`, the strategy generalized through time; if it dies, the in-sample
result was curve-fit (`walk_forward.py` module docstring).

Each fold carries the same purge/embargo hygiene (`purge_days`, `embargo_days`,
`walk_forward.py:136-138`) and reports **Walk-Forward Efficiency (WFE)** — Pardo's named
ratio of *annualized OOS return / annualized IS return* (`_wfe`, `:47`). WFE ≈ 50–80% is
healthy; below ~30% is fragile, above 100% is "too good to be true" (usually a bug or luck).

crucible's **DURABLE** gate hardens this against a specific trap (`fold_dispersion` in
[`validation/diagnostics.py`](../src/crucible/validation/diagnostics.py)): a healthy *average*
WFE can hide individually chaotic folds, so it adds a **fold-dispersion** check — what fraction
of folds are individually tradable (SQN > 0) and the coefficient of variation of fold SQN. High
dispersion is itself a rejection, independent of the average. (pqf's Stage 5 layers its ML-only
IC / feature-stability checks on top of the same crucible diagnostics.)

> **Sources.**
> - **Pardo (2008)** — the walk-forward method and the **Walk-Forward Efficiency** metric
>   are his; the WFA chapter is the primary source. `walk_forward.py` is a capital-free
>   reimplementation of it.
> - **AFML Ch. 12 "Backtesting through Cross-Validation," §12.2 "The Walk-Forward Method"**
>   and **§12.4 "The Combinatorial Purged Cross-Validation Method"** — the modern critique
>   and generalization of single-path walk-forward.
> - SSML Introduction, **"Walkforward Testing"** and **"Overlap Considerations."**

---

## 10. Correcting for correlation: effective sample size (crucible) & portfolio survivability (framework)

A book of 665 trades across 20 markets is not 665 independent bets — eight currency futures
are roughly one dollar bet. Two tools account for this: one **capital-free** and native to
crucible, one **capital-aware** and left to the framework. Both bear directly on whether a
significance claim is honest.

**Effective N** — [`breadth.py`](../src/crucible/breadth.py): `effective_n(returns)` returns a
`Breadth` whose `n_eff = (Σλ)² / Σλ²` is the **participation ratio** of the eigenvalues of the
return-correlation matrix (`participation_ratio`). N perfectly independent markets give
`N_eff = N`; perfectly correlated give `N_eff = 1`. It is *"the honest denominator for
significance"* — a permutation p-value computed as if trades were independent is optimistic
when they cluster into a few factors, which the returned PCA `loadings` then name (dollar /
rates / grains / …). Capital-free: correlation structure only, no equity curve.

```python
>>> from crucible.breadth import effective_n
>>> effective_n(returns).n_eff     # 20-market book, ~3 correlated blocs + a lone metal
3.8                                # ...so it's really ~4 independent bets
```

**Portfolio Monte Carlo** —
[`pardo .../validation/portfolio_mc.py`](../../pardo_quant_framework/src/validation/portfolio_mc.py):
the capital-aware sibling, left in the framework because it needs a capital model crucible
deliberately doesn't have. A **circular block bootstrap** of the monthly portfolio-return
series (`block_bootstrap`, `portfolio_mc.py:49`): contiguous blocks preserve within-period
clustering and autocorrelation that a naive per-trade shuffle destroys; the output is a
distribution of max drawdown, terminal equity, and risk-of-ruin per risk fraction. So
`crucible.breadth` measures the independence *structure*; `portfolio_mc` measures the drawdown
*consequence* of it — the SURVIVE handoff of §11.

> **Sources.**
> - Concurrency / overlap and effective sample size: **AFML Ch. 4 "Sample Weights," §4.3
>   "Number of Concurrent Labels" and §4.4 "Average Uniqueness of a Label"**; correlation-
>   based structure in **Ch. 16 "Machine Learning Asset Allocation"** (HRP; §16.A.1
>   "Correlation-Based Metric").
> - Number of *independent bets* and the **diversification multiplier**: **Carver,
>   *Systematic Trading*, Ch. 11 "Portfolios" and Appendix D "Framework Details →
>   Calculation of diversification multiplier"**; extended in **Carver, *Advanced Futures
>   Trading Strategies*** (instrument diversification and its multiplier across a large
>   universe).
> - Monte Carlo drawdown / risk-of-ruin on reshuffled trade sequences: **Pardo (2008),
>   Monte Carlo and money-management material**; risk of ruin and strategy failure: **AFML
>   Ch. 15 "Understanding Strategy Risk," §15.4 "The Probability of Strategy Failure."**

---

## 11. The whole pipeline, as one gate — `crucible.validation.run_gauntlet`

Every primitive above answers one question. The **gauntlet** runs them as an ordered set
of audited hard gates and returns a single, capital-free pass/fail — crucible's own
naming, no stage numbers borrowed from anywhere:

```
DECLARE   preamble  — a mechanical rule + a log of every variant you tried
CLEAN     preamble  — leakage-controlled construction (use holdout / walk_forward)
──────────────── the gauntlet crucible computes ────────────────
REAL      — distinguishable from noise, corrected for the search
STRONG    — economically meaningful at the CI lower bound
DURABLE   — holds out-of-sample over time
GENERAL   — travels to markets it wasn't built on (optional)
─────────────────────────────────────────────────────────────────
SURVIVE   handoff   — capital survivability (out of scope; hand the surviving log off)
```

```python
from crucible.validation import run_gauntlet

gauntlet = run_gauntlet(
    wf.stitched,        # the honest log — stitched out-of-sample
    prices=px,          # enables REAL's random-entry null
    wf=wf,              # adds the DURABLE gate
    n_variants=4,       # size of your search -> REAL's Šidák correction
)
print(gauntlet.audit_report())
print(gauntlet.passed)  # True only if every gate that ran passed
```

| Gate | Proves | Built from |
|---|---|---|
| **DECLARE** *(preamble)* | the rule is mechanical; the search is logged | EBTA Ch. 1 (§0); the variant count feeds §5 |
| **CLEAN** *(preamble)* | no look-ahead | purge/embargo — §8; the look-ahead-free simulator — §1 |
| **REAL** | not noise, corrected for the search | permutation + Šidák / White's Reality Check — §5; random-entry null — §6 |
| **STRONG** | economically real at the **CI lower bound** | edge metrics — §2, bootstrap CIs — §3 |
| **DURABLE** | survives IS → OOS over time | walk-forward + WFE + fold dispersion — §9 |
| **GENERAL** | travels across markets | cross-market Reality Check — §5; breadth / N_eff — §10 |
| **SURVIVE** *(handoff)* | capital can trade it | **out of scope** — position sizing, drawdown, ruin |

Each gate is an audited AND of its hard checks — a failing hard check can't be waived, and
a strong later gate can't redeem an early failure. The non-negotiable rule: **a FAIL sends
you back to DECLARE, never to tweaking the failing number.** That is the anti-data-mining
discipline made procedural. Full write-up in
[`docs/edge_gate.md`](edge_gate.md).

**Where `pardo_quant_framework` extends it.** crucible stops at "the edge is real, strong,
durable, and general." pqf wraps the same gauntlet in a heavier, capital-aware pipeline
([`edge_validation_framework.md`](../../pardo_quant_framework/docs/edge_validation_framework.md)):
it layers on the **ML-only diagnostics** (IC sign-stability, feature-importance stability),
a **detrended benchmark** (a fractional-return cousin of the random-entry null), the
**cross-asset universe orchestration** behind GENERAL, and — past crucible's boundary — the
**SURVIVE** stage itself: portfolio Monte Carlo, MAR, and correlation on a real capital
model (§10). Same gates, more machinery around them.

---

## Bibliography

Listed in rough order of contribution to the significance machinery. Where a book is
organized by topic rather than fixed pagination (SSML) or where I cite a chapter rather than
a verified page, that is stated explicitly.

1. **Aronson, David R. — *Evidence-Based Technical Analysis*** (Wiley, 2006/2007).
   *The statistical backbone.* Ch. 4 "Statistical Analysis" (p. 165); Ch. 5 "Hypothesis
   Tests and Confidence Intervals" (p. 217) → bootstrap CIs, hypothesis testing; **Ch. 6
   "Data-Mining Bias: The Fool's Gold of Objective TA" (p. 255)** → the permutation test,
   White's Reality Check, multiple comparisons; Ch. 8 "Case Study of Rule Data Mining for
   the S&P 500" (p. 389) → the method applied to 6,402 rules; Appendix "Proof That
   Detrending Is Equivalent to Benchmarking Based on Position Bias" (p. 475) → the detrended
   benchmark. Acknowledgments (p. ix) credit Timothy Masters with the Monte Carlo
   permutation method.

2. **Aronson, David R. & Masters, Timothy — *Statistically Sound Machine Learning for
   Algorithmic Trading of Financial Instruments*** (2013). *Applied companion to EBTA,
   organized by topic around the TSSB tool.* Introduction → "Walkforward Testing," "Cross
   Validation," "Overlap Considerations," "Performance Criteria," "Model Performance Versus
   Financial Performance," "Financial Relevance and Generalizability"; plus the permutation-
   test / selection-bias sections. Source for the sign-permutation implementation and the
   model-vs-financial-performance distinction.

3. **López de Prado, Marcos — *Advances in Financial Machine Learning*** (Wiley, 2018).
   *The leakage-control and ML-labeling backbone.* Ch. 3 "Labeling" §3.4 "The Triple-Barrier
   Method," §3.6–3.7 "Meta-Labeling"; Ch. 4 "Sample Weights" §4.3–4.4 (concurrency,
   uniqueness → effective sample); **Ch. 7 "Cross-Validation in Finance" §7.4 "Purged K-Fold
   CV"** (purge & embargo); Ch. 11 "The Dangers of Backtesting"; Ch. 12 "Backtesting through
   Cross-Validation" §12.2 (walk-forward), §12.4 (CPCV); Ch. 14 "Backtest Statistics"; Ch. 15
   "Understanding Strategy Risk" §15.4 "The Probability of Strategy Failure"; Ch. 16 "ML Asset
   Allocation" (correlation structure).

4. **Pardo, Robert — *The Evaluation and Optimization of Trading Strategies*, 2nd ed.**
   (Wiley, 2008). *The temporal-robustness backbone.* The "Evaluation of Trading Strategies"
   chapter → the classic performance metrics (`edge_report`); the "Walk-Forward Analysis"
   chapter → walk-forward method and the **Walk-Forward Efficiency (WFE)** metric; the
   optimization/robustness and Monte-Carlo money-management material. *(Cited by chapter;
   the scanned copy's page numbers were not individually verified here.)*

5. **Carver, Robert — *Systematic Trading*** (Harriman House, 2015). *Overfitting discipline
   and portfolio structure.* Ch. 3 "Fitting" → in-sample/out-of-sample, robust simple
   parameters; Ch. 9 "Volatility Targeting," Ch. 10 "Position Sizing" → risk-normalized
   returns; Ch. 11 "Portfolios" and Appendix C "Portfolio Optimisation" (bootstrapping,
   rule-of-thumb correlations) and Appendix D "→ Calculation of diversification multiplier"
   → the number-of-independent-bets idea behind `N_eff`.

6. **Carver, Robert — *Advanced Futures Trading Strategies*** (Harriman House, 2021).
   *Referenced to a lesser extent.* The Sharpe-ratio / return-evaluation chapters (a
   performance statistic is itself an estimate with error) and the instrument-diversification
   / diversification-multiplier material across a large futures universe — the practical
   counterpart to `crucible.breadth` and `portfolio_mc.py`. *(Cited at concept/chapter level.)*

**Supporting reference (not in the provided set):**
Van Tharp, Van K. — *Trade Your Way to Financial Freedom*, 2nd ed. (McGraw-Hill, 2007) →
the R-multiple and the **System Quality Number (SQN)**, implemented in `metrics.py:60`.

White, Halbert (2000), "A Reality Check for Data Snooping," *Econometrica* 68(5): 1097–1126 →
the original Reality Check that `whites_reality_check` reimplements via sign permutation.

---

## Code map (where each technique lives)

| Technique | File |
|---|---|
| Trade-log schema, R-multiples | `crucible/src/crucible/edge/trade_log.py` |
| Look-ahead-free barrier simulator | `crucible/src/crucible/edge/simulator.py` |
| Edge metrics (expectancy, PF, SQN, excursion…) | `crucible/src/crucible/edge/metrics.py` |
| Bootstrap CI, p-value, reality_check verdict, random-entry null | `crucible/src/crucible/edge/stats.py` |
| Sign-permutation, Šidák, White's Reality Check | `crucible/src/crucible/validation/permutation.py` |
| Bootstrap metric-set CIs | `crucible/src/crucible/edge/stats.py` |
| PBO (CSCV) + deflated Sharpe | `crucible/src/crucible/validation/pbo.py` |
| ML signal quality — IC, decay, redundancy, PIT | `crucible/src/crucible/ml/` |
| Purged/embargoed holdout | `crucible/src/crucible/validation/holdout.py` |
| Walk-forward + WFE | `crucible/src/crucible/validation/walk_forward.py` |
| Fold dispersion / WFE diagnostics | `crucible/src/crucible/validation/diagnostics.py` |
| Audited gate + Gauntlet | `crucible/src/crucible/validation/gate.py` |
| The gauntlet (REAL/STRONG/DURABLE/GENERAL) + Thresholds | `crucible/src/crucible/validation/gauntlet.py` |
| Effective N / factor PCA | `crucible/src/crucible/breadth.py` |
| Detrended random-timing benchmark | `pardo_quant_framework/src/validation/benchmark.py` |
| Portfolio Monte Carlo (block bootstrap) | `pardo_quant_framework/src/validation/portfolio_mc.py` |
| Triple-barrier labels | `pardo_quant_framework/src/ml/labels.py` |
| Meta-labeling harness | `pardo_quant_framework/src/ml/meta_eval.py` |
| Staged-gate adapter (ML diagnostics + capital stage over crucible) | `pardo_quant_framework/src/validation/stage_evaluator.py` |
| The gated framework, in prose | `pardo_quant_framework/docs/edge_validation_framework.md` |
