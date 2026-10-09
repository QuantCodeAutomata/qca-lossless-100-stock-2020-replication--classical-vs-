"""
Sequential eigenvalue-spacing test for the number of clusters K.

Form the standardized residual matrix X (shape w x N), compute eigenvalues
lambda_1 >= ... >= lambda_N of X.T @ X / w (equivalently, residual correlation
matrix). Compute spacings s_k = lambda_k - lambda_{k+1}. Generate B
column-wise permutation null replicates; obtain the (1 - alpha) quantile q of
null top spacings. Set tau = q * N^{-2/3}. K-hat is the largest k such that
s_1, ..., s_k > tau. If K-hat = 0, use a documented fallback (default K = 2).

# Custom - Context7 found no library equivalent (paper methodology).
"""

from __future__ import annotations

import numpy as np

from .correlation import standardize


def _top_spacings(X: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues (descending) of the correlation matrix of X."""
    w = X.shape[0]
    C = (X.T @ X) / w
    C = 0.5 * (C + C.T)
    vals = np.linalg.eigvalsh(C)[::-1]
    spacings = vals[:-1] - vals[1:]
    return spacings


def estimate_k(
    res_window: np.ndarray,
    alpha: float = 0.05,
    B: int = 200,
    rng: np.random.Generator | None = None,
    K_fallback: int = 2,
) -> int:
    """Estimate the number of clusters K from a standardized residual window."""
    rng = rng or np.random.default_rng(0)
    X = standardize(np.asarray(res_window, dtype=float))
    w, N = X.shape
    s_obs = _top_spacings(X)

    # Permutation null: shuffle each column independently.
    null_max = np.empty(B, dtype=float)
    for b in range(B):
        Xp = X.copy()
        for j in range(N):
            rng.shuffle(Xp[:, j])
        null_max[b] = _top_spacings(Xp).max()
    q = float(np.quantile(null_max, 1.0 - alpha))
    tau = q * (N ** (-2.0 / 3.0))

    K_hat = 0
    for k in range(len(s_obs)):
        if s_obs[k] > tau:
            K_hat = k + 1
        else:
            break
    if K_hat == 0:
        return K_fallback
    return int(min(K_hat, max(2, N - 1)))
