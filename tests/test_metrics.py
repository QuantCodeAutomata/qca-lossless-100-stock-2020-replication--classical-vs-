"""Unit tests for metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import (
    jaccard, mean_jaccard_persistence, sharpe_ratio, sortino_ratio,
    summarize_run, total_return,
)


def test_total_return_zero_for_empty_series() -> None:
    assert total_return(pd.Series(dtype=float)) == 0.0


def test_total_return_simple() -> None:
    s = pd.Series([0.1, -0.05, 0.02])
    tr = total_return(s)
    expected = (1.1 * 0.95 * 1.02) - 1.0
    assert abs(tr - expected) < 1e-12


def test_sharpe_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(1e-3, 1e-2, 500))
    assert sharpe_ratio(s) > 0.0


def test_sharpe_zero_for_constant_series() -> None:
    s = pd.Series([0.001] * 100)
    assert sharpe_ratio(s) == 0.0


def test_sortino_only_penalizes_downside() -> None:
    s = pd.Series([0.01, 0.02, -0.005, 0.015, -0.002])
    assert sortino_ratio(s) > 0.0


def test_sortino_zero_when_no_negatives() -> None:
    s = pd.Series([0.01, 0.02, 0.005])
    assert sortino_ratio(s) == 0.0


def test_jaccard_basic() -> None:
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 2 / 4
    assert jaccard(set(), set()) == 1.0


def test_mean_jaccard_persistence_identical_partitions() -> None:
    p = [[["A", "B"], ["C", "D"]], [["A", "B"], ["C", "D"]]]
    assert mean_jaccard_persistence(p) == 1.0


def test_summarize_run_shape() -> None:
    s = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    out = summarize_run(s, [0.3, 0.4], [1.0, 1.5], [[["A", "B"], ["C"]]])
    assert set(out.keys()) == {"TR", "Sharpe", "Sortino", "WD", "V", "Jaccard"}
    assert all(isinstance(v, float) for v in out.values())
