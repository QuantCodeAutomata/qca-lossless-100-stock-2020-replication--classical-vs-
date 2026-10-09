"""Tests for residual-correlation construction and RMT cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.correlation import gbs_adjacency, residual_corr, rmt_clean, standardize


def test_standardize_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, (20, 10))
    Z = standardize(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(Z.std(axis=0, ddof=0), 1.0, atol=1e-10)


def test_residual_corr_symmetric_unit_diagonal() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 0.01, (5, 8))
    df = pd.DataFrame(X, columns=[f"S{i}" for i in range(8)])
    C = residual_corr(df)
    assert C.shape == (8, 8)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1.0)


def test_rmt_clean_returns_valid_correlation() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(0.0, 0.01, (5, 10))
    df = pd.DataFrame(X, columns=[f"S{i}" for i in range(10)])
    C = residual_corr(df)
    Cc = rmt_clean(C, w=5)
    assert Cc.shape == C.shape
    assert np.allclose(np.diag(Cc), 1.0, atol=1e-6)
    assert np.allclose(Cc, Cc.T, atol=1e-10)


def test_gbs_adjacency_properties() -> None:
    C = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    A = gbs_adjacency(C)
    # Zero diagonal
    assert np.allclose(np.diag(A), 0.0)
    # Non-negative off-diagonals
    assert (A >= 0.0).all()
    # Negative correlation truncated to zero
    assert A[0, 2] == 0.0
    # Symmetric
    assert np.allclose(A, A.T)
