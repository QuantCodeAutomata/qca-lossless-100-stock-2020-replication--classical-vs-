"""
Performance and graph-structure metrics.

* Total return, Sharpe ratio, Sortino ratio.
* Weighted density and cluster value aggregated over rebalance dates.
* Jaccard similarity: for each cluster at t+1, compute max overlap with
  clusters at t and average across clusters.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


ANN_FACTOR = 252.0


def total_return(daily: pd.Series) -> float:
    """Compounded total return of a daily return series."""
    if daily.empty:
        return 0.0
    return float((1.0 + daily).prod() - 1.0)


def sharpe_ratio(daily: pd.Series, rf_daily: float = 0.0) -> float:
    if daily.empty:
        return 0.0
    excess = daily - rf_daily
    sd = excess.std(ddof=0)
    # Treat near-zero volatility (floating-point noise on constant input) as
    # exactly zero so that a flat return series produces a Sharpe of 0 rather
    # than a diverging numerical artefact.
    if sd <= 1e-15 or math.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * math.sqrt(ANN_FACTOR))


def sortino_ratio(daily: pd.Series, rf_daily: float = 0.0) -> float:
    if daily.empty:
        return 0.0
    excess = daily - rf_daily
    neg = excess[excess < 0.0]
    if len(neg) == 0:
        return 0.0
    dd = math.sqrt((neg ** 2).mean())
    if dd == 0.0:
        return 0.0
    return float(excess.mean() / dd * math.sqrt(ANN_FACTOR))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))


def mean_jaccard_persistence(
    partitions: list[list[list[str]]]
) -> float:
    """Average cluster persistence across consecutive rebalance snapshots."""
    if len(partitions) < 2:
        return 0.0
    totals: list[float] = []
    for p_prev, p_next in zip(partitions[:-1], partitions[1:]):
        sets_prev = [set(c) for c in p_prev]
        sets_next = [set(c) for c in p_next]
        if not sets_next or not sets_prev:
            continue
        per_cluster = [max(jaccard(cn, cp) for cp in sets_prev) for cn in sets_next]
        totals.append(float(np.mean(per_cluster)))
    return float(np.mean(totals)) if totals else 0.0


def summarize_run(
    daily: pd.Series,
    wd_series: list[float],
    cv_series: list[float],
    partitions: list[list[list[str]]],
    rf_daily: float = 0.0,
) -> dict[str, float]:
    """Compute summary metrics for one algorithm/seed run."""
    return {
        "TR": total_return(daily),
        "Sharpe": sharpe_ratio(daily, rf_daily),
        "Sortino": sortino_ratio(daily, rf_daily),
        "WD": float(np.mean(wd_series)) if wd_series else 0.0,
        "V": float(np.mean(cv_series)) if cv_series else 0.0,
        "Jaccard": mean_jaccard_persistence(partitions),
    }
