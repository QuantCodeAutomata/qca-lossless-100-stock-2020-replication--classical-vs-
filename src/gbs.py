"""
Gaussian Boson Sampling (GBS) utilities and clustering methods.

Uses Strawberry Fields (`strawberryfields.apps.sample.sample`) for threshold
GBS sampling with optional photon loss. The sampling call signature is
``sample.sample(A, n_mean, n_samples, threshold=True, loss=0.0)``.

Clustering methods implemented from the paper's methodology:
  * ``gbs_boost``: iteratively select the densest sample subgraph (unweighted
    density), remove it, repeat.
  * ``gbs_roots``: generate samples, sort by weighted density, build a disjoint
    union greedily, extract.
  * ``qic_gbs``: quantum-inspired comparison method using a reduced sample
    budget H = round(N log N); extraction uses weighted density similar to
    gbs_roots but without the greedy union step.

Residual-graph termination rule (shared across all methods): while the
residual graph has more than 2 nodes, sample and extract. If 2 nodes remain,
assign them as the final cluster.  If an extraction would leave a singleton,
assign the entire residual graph as the final cluster.

# Using Strawberry Fields threshold GBS sampling - Context7 confirmed.
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Backend selection -----------------------------------------------------------
# ``GBS_BACKEND`` env var picks between a fast surrogate sampler ("fast") and
# the exact Strawberry Fields threshold-GBS sampler ("sf").  The exact SF
# backend is numerically expensive (Torontonian computation is #P-hard), which
# makes it intractable for the N ~ 50-100 graphs and multi-seed rolling
# experiments of this paper.  The fast backend preserves the two features that
# matter for the paper's clustering methods:
#   1. Sampled subgraphs are biased toward high weighted-degree nodes, i.e.
#      denser regions of the graph have higher sampling probability.
#   2. Photon loss attenuates the "effective nbar" so sampled subgraphs shrink
#      and become less discriminative as the loss rate rises.
# Both properties follow from the GBS probability of a click pattern being
# proportional to |Hafnian(A_S)|^2 for threshold detection, with an eta
# scaling under uniform loss.  We approximate this via a greedy weighted-edge
# chain that produces dense subgraphs with the same qualitative scaling in
# loss and nbar.  This is a documented approximation; see RESULTS.md.
_BACKEND = os.environ.get("GBS_BACKEND", "fast").lower()


def _fast_threshold_samples(
    A: np.ndarray,
    H: int,
    loss: float = 0.0,
    nbar: float | None = None,
    rng: np.random.Generator | None = None,
    displace: bool = False,
) -> list[list[int]]:
    """Fast GBS-like threshold sampler (documented approximation).

    For each sample we draw a target click count k ~ clipped Poisson around
    the effective mean photon number sqrt(1-loss) * nbar, then build a dense
    subgraph of size k by weighted random walks on A.  Nodes with higher
    weighted degree are more likely to appear, and dense subgraphs are more
    likely to co-occur -- matching the GBS density bias.

    When ``displace`` is True, the effective mean photon number is held at
    ``nbar`` regardless of loss, modelling coherent displacement compensation.
    This is the surrogate rule used in the lossy experiment (exp 2) and is
    marked as an approximation in RESULTS.md.
    """
    rng = rng or np.random.default_rng()
    N = A.shape[0]
    nbar = math.sqrt(max(N, 1)) if nbar is None else float(nbar)
    eta = max(0.0, 1.0 - float(loss))
    # Displacement surrogate: compensate the mean-photon-number loss.
    eff_nbar = nbar if displace else math.sqrt(eta) * nbar
    deg = A.sum(axis=1)
    tot = deg.sum()
    if tot <= 0:
        return []
    p0 = deg / tot
    samples: list[list[int]] = []
    for _ in range(max(1, int(H))):
        # Target click count: clipped Poisson around eff_nbar; small floor of 2.
        k = int(np.clip(rng.poisson(max(eff_nbar, 1.0)), 2, max(2, N - 1)))
        # Seed node from weighted-degree distribution.
        current = int(rng.choice(N, p=p0))
        clicks = {current}
        for _step in range(k - 1):
            # Weighted neighbors of the current cluster:
            w = A[list(clicks), :].sum(axis=0)
            w[list(clicks)] = 0.0
            s = w.sum()
            if s <= 0:
                break
            nxt = int(rng.choice(N, p=w / s))
            clicks.add(nxt)
        if len(clicks) < 2:
            continue
        pat = [0] * N
        for v in clicks:
            pat[v] = 1
        samples.append(pat)
    return samples


def _sf_threshold_samples(
    A: np.ndarray, H: int, loss: float, nbar: float,
) -> list[list[int]]:
    """Exact Strawberry Fields threshold-GBS sampler (slow)."""
    from strawberryfields.apps import sample as sf_sample
    raw = sf_sample.sample(
        A.astype(float), n_mean=float(nbar), n_samples=max(1, int(H)),
        threshold=True, loss=float(loss),
    )
    return [list(map(int, s)) for s in raw if sum(s) >= 2]


def gbs_threshold_samples(
    A: np.ndarray,
    H: int,
    loss: float = 0.0,
    nbar: float | None = None,
    rng: np.random.Generator | None = None,
    displace: bool = False,
) -> list[list[int]]:
    """Draw H threshold-detection samples from GBS with adjacency A.

    Dispatches to a fast approximate sampler (default) or the exact Strawberry
    Fields threshold sampler, selected via the ``GBS_BACKEND`` env var.
    Samples with fewer than 2 clicks are discarded.

    If ``displace`` is True, the lossy sampler uses coherent-displacement
    compensation so that the effective mean photon number is approximately
    preserved under loss.  This is the surrogate rule used in exp 2.
    """
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    N = A.shape[0]
    if nbar is None:
        nbar = math.sqrt(max(N, 1))
    if A.sum() == 0.0:
        return []
    if _BACKEND == "sf":
        # The SF applications sampler does not expose a displacement knob;
        # fall back to the fast sampler whenever displacement is requested.
        if displace:
            return _fast_threshold_samples(
                A, H, float(loss), float(nbar), rng=rng, displace=True
            )
        return _sf_threshold_samples(A, H, float(loss), float(nbar))
    return _fast_threshold_samples(
        A, H, float(loss), float(nbar), rng=rng, displace=displace
    )


def clicks_to_nodes(clicks: list[int]) -> list[int]:
    """Convert a click pattern to the set of node indices with click = 1."""
    return [i for i, c in enumerate(clicks) if c == 1]


# ---------------------------------------------------------------------------
# Graph metrics on subgraphs
# ---------------------------------------------------------------------------
def unweighted_density(A: np.ndarray, S: list[int]) -> float:
    """Density of the unweighted subgraph induced by S (ignoring edge weights)."""
    n = len(S)
    if n < 2:
        return 0.0
    sub = A[np.ix_(S, S)]
    edges = (sub > 0).sum() / 2
    return float(edges) / (n * (n - 1) / 2)


def weighted_density(A: np.ndarray, S: list[int]) -> float:
    """WD(S) = sum_{i<j in S} A_ij / [|S|(|S|-1)/2]."""
    n = len(S)
    if n < 2:
        return 0.0
    sub = A[np.ix_(S, S)]
    tot = sub.sum() / 2.0
    return float(tot) / (n * (n - 1) / 2)


def cluster_value(C: np.ndarray, S: list[int]) -> float:
    """Within-cluster pairwise correlation sum (uses signed cleaned matrix)."""
    n = len(S)
    if n < 2:
        return 0.0
    sub = C[np.ix_(S, S)]
    tot = sub.sum() / 2.0
    # Remove the diagonal half-count (self-correlation = 1).
    tot -= 0.5 * np.trace(sub)
    return float(tot)


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------
def _finalize_partition(partition: list[list[int]], N: int) -> list[list[int]]:
    """Ensure the partition is exhaustive and disjoint over range(N)."""
    seen: set[int] = set()
    out: list[list[int]] = []
    for cl in partition:
        cl2 = [i for i in cl if i not in seen]
        if cl2:
            out.append(sorted(cl2))
            seen.update(cl2)
    missing = sorted(set(range(N)) - seen)
    if missing:
        out.append(missing)
    return out


def _loop_extract(
    A: np.ndarray,
    sample_fn,
    extract_fn,
) -> list[list[int]]:
    """Shared skeleton: iteratively sample & extract clusters until residual
    graph has <= 2 nodes. Returns a partition of range(A.shape[0])."""
    N = A.shape[0]
    active = list(range(N))
    partition: list[list[int]] = []
    safety = 0
    while len(active) > 2 and safety < 2 * N:
        safety += 1
        A_sub = A[np.ix_(active, active)]
        samples = sample_fn(A_sub)
        if not samples:
            # No usable samples: dump the residual graph as a final cluster.
            partition.append(active[:])
            active = []
            break
        chosen_local = extract_fn(A_sub, samples)
        if not chosen_local:
            partition.append(active[:])
            active = []
            break
        chosen_global = sorted(active[i] for i in chosen_local)
        remaining = [v for v in active if v not in set(chosen_global)]
        # Termination rules.
        if len(remaining) == 1:
            partition.append(sorted(chosen_global + remaining))
            active = []
            break
        partition.append(chosen_global)
        active = remaining
    if active:
        partition.append(sorted(active))
    return _finalize_partition(partition, N)


# --- GBS Boost: unweighted density, pick densest sample ---------------------
def gbs_boost_cluster(
    A: np.ndarray, H_factor: float = 0.7, loss: float = 0.0,
    rng: np.random.Generator | None = None, displace: bool = False,
) -> list[list[int]]:
    """GBS Boost clustering (paper Algorithm 1)."""
    def sampler(A_sub):
        N = A_sub.shape[0]
        H = max(1, round(H_factor * N * math.log(max(N, 2))))
        return gbs_threshold_samples(
            A_sub, H, loss=loss, rng=rng, displace=displace
        )

    def extractor(A_sub, samples):
        # Score by unweighted density; tie-break by weighted density.
        best = None
        best_score = (-np.inf, -np.inf)
        for s in samples:
            nodes = clicks_to_nodes(s)
            if len(nodes) < 2:
                continue
            score = (unweighted_density(A_sub, nodes), weighted_density(A_sub, nodes))
            if score > best_score:
                best_score = score
                best = nodes
        return best or []

    return _loop_extract(A, sampler, extractor)


# --- GBS Roots: weighted density + greedy disjoint union --------------------
def gbs_roots_cluster(
    A: np.ndarray, H_factor: float = 0.6, loss: float = 0.0,
    rng: np.random.Generator | None = None, displace: bool = False,
) -> list[list[int]]:
    """GBS Roots clustering (paper Algorithm 2)."""
    def sampler(A_sub):
        N = A_sub.shape[0]
        H = max(1, round(H_factor * N * math.log(max(N, 2))))
        return gbs_threshold_samples(
            A_sub, H, loss=loss, rng=rng, displace=displace
        )

    def extractor(A_sub, samples):
        scored = []
        for s in samples:
            nodes = clicks_to_nodes(s)
            if len(nodes) < 2:
                continue
            scored.append((weighted_density(A_sub, nodes), nodes))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[0])
        U = list(scored[0][1])
        wd_U = weighted_density(A_sub, U)
        U_set = set(U)
        for wd, nodes in scored[1:]:
            if any(n in U_set for n in nodes):
                continue  # Disjointness required.
            candidate = U + nodes
            wd_cand = weighted_density(A_sub, candidate)
            if wd_cand > wd_U:
                U = candidate
                U_set.update(nodes)
                wd_U = wd_cand
            else:
                break  # Stop when candidate fails to improve WD.
        return U

    return _loop_extract(A, sampler, extractor)


# --- QIC-GBS: quantum-inspired comparison with reduced budget ---------------
def qic_gbs_cluster(
    A: np.ndarray, H_factor: float = 1.0, loss: float = 0.0,
    rng: np.random.Generator | None = None, displace: bool = False,
) -> list[list[int]]:
    """QIC-GBS clustering: H = round(F * N log N), extract densest WD sample."""
    def sampler(A_sub):
        N = A_sub.shape[0]
        H = max(1, round(H_factor * N * math.log(max(N, 2))))
        return gbs_threshold_samples(
            A_sub, H, loss=loss, rng=rng, displace=displace
        )

    def extractor(A_sub, samples):
        best = None
        best_wd = -np.inf
        for s in samples:
            nodes = clicks_to_nodes(s)
            if len(nodes) < 2:
                continue
            wd = weighted_density(A_sub, nodes)
            if wd > best_wd:
                best_wd = wd
                best = nodes
        return best or []

    return _loop_extract(A, sampler, extractor)


# --- Random baseline (matches cluster-size distribution) --------------------
def random_cluster(
    A: np.ndarray, cluster_sizes: list[int], rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Random partition with the given cluster sizes. `cluster_sizes` must
    sum to A.shape[0]; mismatches are repaired by largest-remainder."""
    rng = rng or np.random.default_rng(0)
    N = A.shape[0]
    sizes = list(cluster_sizes)
    diff = N - sum(sizes)
    if diff > 0:
        sizes[-1] += diff
    elif diff < 0:
        for i in range(len(sizes) - 1, -1, -1):
            take = min(-diff, sizes[i] - 1)
            sizes[i] -= take
            diff += take
            if diff == 0:
                break
    perm = rng.permutation(N).tolist()
    partition: list[list[int]] = []
    pos = 0
    for s in sizes:
        partition.append(sorted(perm[pos: pos + s]))
        pos += s
    return _finalize_partition(partition, N)
