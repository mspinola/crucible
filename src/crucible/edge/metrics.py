"""Capital-free edge metrics. Every function takes plain arrays; `edge_report`
assembles the full scorecard from whatever columns a TradeLog carries.

Lineage of the less-obvious ones:
  excursion_ratio / e_ratio  — MFE/MAE efficiency (Sweeney; Faith, "Way of the Turtle")
  sqn                        — System Quality Number (Van Tharp)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from crucible.edge.trade_log import TradeLog


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def win_rate(r) -> float:
    r = _arr(r)
    return float((r > 0).mean()) if len(r) else 0.0


def expectancy(r) -> float:
    """Normalized expectancy: win_rate*avg_win − loss_rate*avg_loss (in R)."""
    r = _arr(r)
    n = len(r)
    if n == 0:
        return 0.0
    wins, losses = r[r > 0], r[r < 0]
    wr, lr = len(wins) / n, len(losses) / n
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    return float(wr * avg_win - lr * avg_loss)


def profit_factor(r) -> float:
    """Gross profit / gross loss. +inf if there are no losers."""
    r = _arr(r)
    gp = r[r > 0].sum()
    gl = abs(r[r < 0].sum())
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


def payoff_ratio(r) -> float:
    """Average win / average loss (terminal risk-reward)."""
    r = _arr(r)
    wins, losses = r[r > 0], r[r < 0]
    if not len(wins) or not len(losses):
        return float("inf") if len(wins) else 0.0
    return float(wins.mean() / abs(losses.mean()))


def sqn(r, cap: int = 100) -> float:
    """Van Tharp System Quality Number: mean/std × √n, n capped (default 100)."""
    r = _arr(r)
    n = len(r)
    if n < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(min(n, cap)))


def excursion_ratio(mfe, mae) -> float:
    """mean(MFE) / |mean(MAE)| — how far trades run for vs. against you."""
    mfe, mae = _arr(mfe), _arr(mae)
    if not len(mfe) or not len(mae):
        return float("nan")
    m = abs(mae.mean())
    return float(mfe.mean() / m) if m else float("inf")


def e_ratio(mfe_k, mae_k) -> float:
    """Edge ratio at a fixed horizon k (Faith): mean(MFE_k)/|mean(MAE_k)|.
    >1 means the signal has directional edge before any exit rule is applied."""
    return excursion_ratio(mfe_k, mae_k)


def time_asymmetry(bars_held, r) -> float:
    """avg bars in winners / avg bars in losers. >1 = let winners run, cut losers."""
    bars_held, r = np.asarray(bars_held, dtype=float), np.asarray(r, dtype=float)
    win_bars = bars_held[r > 0]
    loss_bars = bars_held[r < 0]
    if not len(win_bars) or not len(loss_bars):
        return float("nan")
    denom = loss_bars.mean()
    return float(win_bars.mean() / denom) if denom else float("inf")


def exit_efficiency(r, mfe) -> float:
    """Among winners: captured return / available MFE. How much of the favorable
    move the exit actually banked (1.0 = sold the top)."""
    r, mfe = np.asarray(r, dtype=float), np.asarray(mfe, dtype=float)
    mask = r > 0
    if not mask.any():
        return float("nan")
    avail = mfe[mask].mean()
    return float(r[mask].mean() / avail) if avail > 0 else float("nan")


def _calibration(prob, r) -> dict:
    """Win rate within model-confidence bins — is prob well-calibrated?"""
    import pandas as pd
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0001]
    labels = ["<50%", "50-60%", "60-70%", "70-80%", "80%+"]
    df = pd.DataFrame({"prob": prob, "r": r}).dropna()
    if df.empty:
        return {}
    df["bin"] = pd.cut(df["prob"], bins=bins, labels=labels, right=False)
    out = {}
    for name, g in df.groupby("bin", observed=True):
        if len(g):
            out[str(name)] = {"trades": int(len(g)),
                              "win_rate": float((g["r"] > 0).mean())}
    return out


@dataclass
class EdgeReport:
    n: int
    win_rate: float
    expectancy: float
    profit_factor: float
    payoff_ratio: float
    sqn: float
    excursion_ratio: Optional[float] = None
    e_ratio: Optional[float] = None
    time_asymmetry: Optional[float] = None
    exit_efficiency: Optional[float] = None
    calibration: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        def flag(ok, warn_only=False):
            return "[PASS]" if ok else ("[WARN]" if warn_only else "[FAIL]")

        L = ["=" * 56, " EDGE REPORT (capital-free)", "=" * 56]
        L.append(f"Trades              : {self.n}")
        L.append(f"Win rate            : {self.win_rate * 100:.1f} %")
        L.append("-" * 56)
        L.append(f"Expectancy          : {self.expectancy:+.3f} R      {flag(self.expectancy > 0)}")
        pf = self.profit_factor
        L.append(f"Profit factor       : {pf:.2f}          {flag(pf > 1.25, warn_only=True)}")
        L.append(f"Payoff ratio        : {self.payoff_ratio:.2f}          [INFO]")
        L.append(f"SQN-100             : {self.sqn:.2f}          [INFO]")
        if self.excursion_ratio is not None:
            L.append("-" * 56)
            L.append(f"Excursion ratio     : {self.excursion_ratio:.2f}          {flag(self.excursion_ratio > 1.0)}")
        if self.e_ratio is not None:
            L.append(f"E-ratio (k-bar)     : {self.e_ratio:.2f}          {flag(self.e_ratio > 1.0, warn_only=True)}")
        if self.time_asymmetry is not None:
            L.append(f"Time asymmetry      : {self.time_asymmetry:.2f}          {flag(self.time_asymmetry > 1.0, warn_only=True)}")
        if self.exit_efficiency is not None:
            L.append(f"Exit efficiency     : {self.exit_efficiency * 100:.1f} %        [INFO]")
        if self.calibration:
            L.append("-" * 56)
            L.append("Calibration (confidence -> win rate):")
            for k, v in self.calibration.items():
                L.append(f"  {k:8s} -> {v['win_rate'] * 100:5.1f}%  (n={v['trades']})")
        L.append("=" * 56)
        return "\n".join(L)


def edge_report(trades: TradeLog) -> EdgeReport:
    """Every capital-free edge metric the TradeLog has the columns to support."""
    r = trades.r
    mfe, mae = trades.col("mfe"), trades.col("mae")
    bars, prob = trades.col("bars_held"), trades.col("prob")
    return EdgeReport(
        n=trades.n,
        win_rate=win_rate(r),
        expectancy=expectancy(r),
        profit_factor=profit_factor(r),
        payoff_ratio=payoff_ratio(r),
        sqn=sqn(r),
        excursion_ratio=excursion_ratio(mfe, mae) if mfe is not None and mae is not None else None,
        e_ratio=None,  # populate when you pass a fixed-horizon MFE/MAE via e_ratio()
        time_asymmetry=time_asymmetry(bars, r) if bars is not None else None,
        exit_efficiency=exit_efficiency(r, mfe) if mfe is not None else None,
        calibration=_calibration(prob, r) if prob is not None else None,
    )
