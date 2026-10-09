"""Tests for the classical spectral and SPONGE baselines."""

from __future__ import annotations

import numpy as np

from src.clustering_classical import (
    labels_to_partition, spectral_clustering, sponge_sym,
)


def _planted_signed_corr(block_sizes: list[int]) -> np.ndarray:
    N = sum(block_sizes)
    C = np.eye(N)
    start = 0
    for s in block_sizes:
        for i in range(start, start + s):
            for j in range(start, start + s):
                if i != j:
                    C[i, j] = 0.7
        start += s
    return C


def test_spectral_partitions_valid_shape() -> None:
    C = _planted_signed_corr([5, 5, 5])
    labels = spectral_clustering(C, K=3, random_state=0)
    assert labels.shape == (15,)
    assert set(labels.tolist()).issubset({0, 1, 2})


def test_spectral_single_cluster_edge_case() -> None:
    C = np.eye(5)
    labels = spectral_clustering(C, K=1)
    assert (labels == 0).all()


def test_sponge_handles_signed_matrix() -> None:
    C = np.eye(10)
    # Positive within-block, negative between-block
    for i in range(5):
        for j in range(5):
            if i != j:
                C[i, j] = 0.6
                C[i + 5, j + 5] = 0.6
    for i in range(5):
        for j in range(5):
            C[i, j + 5] = -0.4
            C[j + 5, i] = -0.4
    labels = sponge_sym(C, K=2, random_state=0)
    assert labels.shape == (10,)
    assert set(labels.tolist()).issubset({0, 1})


def test_labels_to_partition_roundtrip() -> None:
    labels = np.array([0, 1, 0, 2, 1])
    parts = labels_to_partition(labels)
    flat = sorted(sum(parts, []))
    assert flat == [0, 1, 2, 3, 4]
    # Expected 3 clusters.
    assert len(parts) == 3
