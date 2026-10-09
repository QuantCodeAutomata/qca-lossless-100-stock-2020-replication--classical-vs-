"""Tests for the GBS sampling and clustering primitives."""

from __future__ import annotations

import numpy as np

from src.gbs import (
    clicks_to_nodes, cluster_value, gbs_boost_cluster, gbs_roots_cluster,
    gbs_threshold_samples, qic_gbs_cluster, random_cluster, unweighted_density,
    weighted_density,
)


def _block_adjacency(block_sizes: list[int], seed: int = 0) -> np.ndarray:
    """Build a planted block-diagonal weighted adjacency (zero diagonal)."""
    rng = np.random.default_rng(seed)
    N = sum(block_sizes)
    A = np.zeros((N, N))
    start = 0
    for s in block_sizes:
        for i in range(start, start + s):
            for j in range(i + 1, start + s):
                A[i, j] = A[j, i] = 0.6 + 0.3 * rng.random()
        start += s
    return A


def test_sampler_returns_lists_of_right_length() -> None:
    A = _block_adjacency([5, 5, 5])
    rng = np.random.default_rng(0)
    samples = gbs_threshold_samples(A, H=10, rng=rng)
    assert len(samples) > 0
    for s in samples:
        assert len(s) == A.shape[0]
        assert sum(s) >= 2


def test_weighted_density_bounds() -> None:
    A = _block_adjacency([5, 5])
    wd_in_block = weighted_density(A, [0, 1, 2, 3, 4])
    wd_across = weighted_density(A, [0, 5])
    assert wd_in_block > wd_across
    assert 0.0 <= wd_in_block <= 1.0


def test_unweighted_density_bounds() -> None:
    A = _block_adjacency([4, 4])
    d = unweighted_density(A, [0, 1, 2, 3])
    assert 0.0 <= d <= 1.0


def test_cluster_value_signed_matrix() -> None:
    C = np.eye(4)
    C[0, 1] = C[1, 0] = 0.9
    C[2, 3] = C[3, 2] = 0.4
    v = cluster_value(C, [0, 1])
    assert abs(v - 0.9) < 1e-10


def test_clicks_to_nodes_roundtrip() -> None:
    assert clicks_to_nodes([1, 0, 1, 1, 0]) == [0, 2, 3]


def test_gbs_boost_partitions_cover_all_nodes() -> None:
    A = _block_adjacency([4, 4, 4])
    rng = np.random.default_rng(0)
    part = gbs_boost_cluster(A, H_factor=1.0, rng=rng)
    flat = sorted(sum(part, []))
    assert flat == list(range(A.shape[0]))


def test_gbs_roots_partitions_cover_all_nodes() -> None:
    A = _block_adjacency([3, 3, 3, 3])
    rng = np.random.default_rng(0)
    part = gbs_roots_cluster(A, H_factor=1.0, rng=rng)
    flat = sorted(sum(part, []))
    assert flat == list(range(A.shape[0]))


def test_qic_gbs_partitions_cover_all_nodes() -> None:
    A = _block_adjacency([4, 4])
    rng = np.random.default_rng(0)
    part = qic_gbs_cluster(A, H_factor=2.0, rng=rng)
    flat = sorted(sum(part, []))
    assert flat == list(range(A.shape[0]))


def test_random_cluster_matches_sizes() -> None:
    A = np.zeros((10, 10))
    rng = np.random.default_rng(0)
    part = random_cluster(A, [3, 3, 4], rng=rng)
    assert sorted(len(c) for c in part) == [3, 3, 4]
    flat = sorted(sum(part, []))
    assert flat == list(range(10))


def test_sampler_shrinks_under_loss() -> None:
    """High photon loss should produce smaller click patterns on average."""
    A = _block_adjacency([6, 6])
    rng = np.random.default_rng(0)
    low_loss = gbs_threshold_samples(A, H=50, loss=0.0, rng=rng)
    high_loss = gbs_threshold_samples(A, H=50, loss=0.9, rng=np.random.default_rng(0))
    avg_low = np.mean([sum(s) for s in low_loss]) if low_loss else 0.0
    avg_high = np.mean([sum(s) for s in high_loss]) if high_loss else 0.0
    assert avg_low >= avg_high
