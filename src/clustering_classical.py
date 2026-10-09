"""
Classical clustering baselines.

* ``spectral_clustering``: Ng-Jordan-Weiss style spectral clustering on the
  cleaned correlation matrix. Uses an affinity transform A' = max(0, C) to
  guarantee non-negative weights.
* ``sponge_sym``: symmetric SPONGE signed-graph clustering, implemented
  directly from the reference equation
      L_sym = (D_pos + tau_n * I)^{-1/2} * (L_pos - rho_sponge * L_neg)
              * (D_pos + tau_n * I)^{-1/2}
  with eigenvectors of L_sym embedded then clustered with k-means.
  # Custom - Context7 confirmed no maintained SPONGE package is available.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


def spectral_clustering(
    C_clean: np.ndarray, K: int, random_state: int = 0
) -> np.ndarray:
    """Spectral clustering on the non-negative part of the cleaned matrix."""
    N = C_clean.shape[0]
    K = max(1, min(K, N))
    if K == 1:
        return np.zeros(N, dtype=int)
    A = np.maximum(0.0, C_clean.copy())
    np.fill_diagonal(A, 0.0)
    # Guarantee at least some edges to avoid degenerate affinity.
    if A.sum() == 0.0:
        return np.zeros(N, dtype=int)
    sc = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    return sc.fit_predict(A)


def sponge_sym(
    C_clean: np.ndarray, K: int, rho: float = 1.0, tau_n: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Symmetric SPONGE clustering on a signed cleaned correlation matrix.

    Reference equation implemented from scratch (Cucuringu et al., 2019).
    """
    N = C_clean.shape[0]
    K = max(1, min(K, N))
    if K == 1:
        return np.zeros(N, dtype=int)
    W = C_clean.copy()
    np.fill_diagonal(W, 0.0)
    Wp = np.maximum(0.0, W)
    Wn = np.maximum(0.0, -W)
    Dp = np.diag(Wp.sum(axis=1))
    Dn = np.diag(Wn.sum(axis=1))
    Lp = Dp - Wp
    Ln = Dn - Wn
    inv_sqrt_Dp = np.diag(1.0 / np.sqrt(np.clip(np.diag(Dp) + tau_n, 1e-12, None)))
    L_sym = inv_sqrt_Dp @ (Lp - rho * Ln) @ inv_sqrt_Dp
    L_sym = 0.5 * (L_sym + L_sym.T)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # Smallest-K eigenvectors as embedding (ascending order).
    emb = eigvecs[:, :K]
    # Row-normalize (Ng-Jordan-Weiss style).
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb = emb / norms
    km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    return km.fit_predict(emb)


def labels_to_partition(labels: np.ndarray) -> list[list[int]]:
    """Convert a label vector into a list of index lists (clusters)."""
    out: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        out.setdefault(int(l), []).append(int(i))
    return [sorted(v) for _, v in sorted(out.items())]
