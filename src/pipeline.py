"""
End-to-end rolling pipeline: residualize, clean correlation, cluster, trade.

A single ``ClusterStrategy`` instance owns:
  * the daily total-return panel for a frozen universe,
  * the Fama-French daily market excess return,
  * the clustering method name and its hyperparameters,
  * per-rebalance diagnostic logging (clusters, WD, V).

The ``run()`` method iterates rebalance dates and returns:
  * ``daily_returns`` -- a pd.Series of realized portfolio daily returns,
  * ``diag`` -- a dict with lists of WD, V, cluster partitions per date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .clustering_classical import (
    labels_to_partition, sponge_sym, spectral_clustering,
)
from .correlation import gbs_adjacency, residual_corr, rmt_clean
from .gbs import (
    cluster_value, gbs_boost_cluster, gbs_roots_cluster, qic_gbs_cluster,
    random_cluster, weighted_density,
)
from .k_selection import estimate_k
from .residuals import estimate_betas, residual_returns
from .trading import (
    aggregate_cluster_weights, make_rebalance_calendar, simulate_trade_pnl,
)


CLUSTERING_METHODS = (
    "spectral", "sponge", "gbs_boost", "gbs_roots", "qic_gbs", "random"
)


@dataclass
class ClusterStrategy:
    ret: pd.DataFrame                # daily total returns, columns = tickers
    mkt: pd.Series                   # Fama-French market excess return
    method: str                      # one of CLUSTERING_METHODS
    beta_window: int = 60
    corr_lookback: int = 5
    holding_period: int = 3
    stop_win: float = 0.02
    rebalance_freq: int = 3
    threshold_p: float = 0.0
    # method-specific hyperparameters
    H_factor: float = 0.7            # for gbs_boost / qic_gbs
    H_factor_roots: float = 0.6      # for gbs_roots
    H_factor_qic: float = 1.0        # for qic_gbs
    loss: float = 0.0                # photon loss (0..1) for GBS methods
    displace: bool = False           # coherent-displacement compensation
    k_B: int = 100                   # permutation reps for K selection
    k_alpha: float = 0.05
    seed: int = 0
    random_sizes: list[int] | None = None   # for random baseline

    # test-year constraints
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None

    # internal state
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        assert self.method in CLUSTERING_METHODS, self.method
        self._rng = np.random.default_rng(self.seed)

    # ---------------------------------------------------------------------
    def _cluster_once(
        self, T: pd.Timestamp, res_window: pd.DataFrame
    ) -> list[list[str]]:
        """Run the clustering method on a single residual-return window."""
        tickers = list(res_window.columns)
        C = residual_corr(res_window)
        C_clean = rmt_clean(C, w=self.corr_lookback)
        if self.method in ("spectral", "sponge"):
            K = estimate_k(
                res_window.to_numpy(),
                alpha=self.k_alpha,
                B=self.k_B,
                rng=self._rng,
            )
            if self.method == "spectral":
                labels = spectral_clustering(C_clean, K, random_state=self.seed)
            else:
                labels = sponge_sym(C_clean, K, random_state=self.seed)
            partition_idx = labels_to_partition(labels)
        elif self.method == "random":
            # Random baseline matches a given cluster-size profile.
            N = C_clean.shape[0]
            sizes = self.random_sizes or [max(2, N // 5)] * 5
            A = gbs_adjacency(C_clean)
            partition_idx = random_cluster(A, sizes, self._rng)
        else:
            A = gbs_adjacency(C_clean)
            if self.method == "gbs_boost":
                partition_idx = gbs_boost_cluster(
                    A, H_factor=self.H_factor, loss=self.loss,
                    rng=self._rng, displace=self.displace,
                )
            elif self.method == "gbs_roots":
                partition_idx = gbs_roots_cluster(
                    A, H_factor=self.H_factor_roots, loss=self.loss,
                    rng=self._rng, displace=self.displace,
                )
            elif self.method == "qic_gbs":
                partition_idx = qic_gbs_cluster(
                    A, H_factor=self.H_factor_qic, loss=self.loss,
                    rng=self._rng, displace=self.displace,
                )
            else:  # pragma: no cover
                raise ValueError(f"Unknown method {self.method}")
        return [[tickers[i] for i in cl] for cl in partition_idx]

    # ---------------------------------------------------------------------
    def _compute_wd_v(
        self, partition_idx: list[list[int]], C_clean: np.ndarray, A: np.ndarray
    ) -> tuple[float, float]:
        wd_list = [weighted_density(A, cl) for cl in partition_idx if len(cl) >= 2]
        v_list = [cluster_value(C_clean, cl) for cl in partition_idx if len(cl) >= 2]
        wd = float(np.mean(wd_list)) if wd_list else 0.0
        v = float(np.sum(v_list)) if v_list else 0.0
        return wd, v

    # ---------------------------------------------------------------------
    def run(self) -> tuple[pd.Series, dict]:
        idx = self.ret.index
        start_pos = self.beta_window + self.corr_lookback + 1
        if self.start_date is not None:
            sp = idx.searchsorted(pd.Timestamp(self.start_date))
            start_pos = max(start_pos, int(sp))
        rebal = make_rebalance_calendar(idx, start_pos, self.rebalance_freq)
        if self.end_date is not None:
            rebal = [d for d in rebal if d <= pd.Timestamp(self.end_date)]
        daily_rets: dict[pd.Timestamp, float] = {}
        wd_series: list[float] = []
        v_series: list[float] = []
        partitions_log: list[list[list[str]]] = []
        n_pos_log: list[int] = []
        i = 0
        while i < len(rebal):
            T = rebal[i]
            T_pos = idx.get_indexer([T])[0]
            if T_pos < self.beta_window + self.corr_lookback:
                i += 1
                continue
            if T_pos + self.holding_period > len(idx):
                break
            # Residualize using info up to T-1.
            try:
                betas = estimate_betas(self.ret, self.mkt, T, self.beta_window)
            except ValueError:
                i += 1
                continue
            res_full = residual_returns(self.ret, self.mkt, betas)
            res_window = res_full.iloc[T_pos - self.corr_lookback: T_pos]
            if res_window.isna().any().any():
                i += 1
                continue
            # Clustering.
            clusters = self._cluster_once(T, res_window)
            partitions_log.append(clusters)
            # Diagnostics on the signed cleaned matrix.
            C = residual_corr(res_window)
            C_clean = rmt_clean(C, w=self.corr_lookback)
            A = gbs_adjacency(C_clean)
            tickers = list(res_window.columns)
            idx_map = {t: j for j, t in enumerate(tickers)}
            partition_idx = [[idx_map[t] for t in cl if t in idx_map] for cl in clusters]
            wd, v = self._compute_wd_v(partition_idx, C_clean, A)
            wd_series.append(wd)
            v_series.append(v)
            # Trading: build weights from the formation window.
            ret_window = self.ret.iloc[T_pos - self.corr_lookback: T_pos]
            weights = aggregate_cluster_weights(clusters, ret_window, self.threshold_p)
            trade_rets = simulate_trade_pnl(
                weights,
                self.ret.iloc[T_pos: T_pos + self.holding_period],
                holding_period=self.holding_period,
                stop_win=self.stop_win,
            )
            n_pos_log.append(len(weights))
            for d, r in enumerate(trade_rets):
                daily_rets[idx[T_pos + d]] = r
            held = len(trade_rets)
            next_pos = T_pos + max(held, 1)
            while i < len(rebal) and idx.get_indexer([rebal[i]])[0] < next_pos:
                i += 1
            if i < len(rebal) and rebal[i] == T:
                i += 1
        s = pd.Series(daily_rets).sort_index() if daily_rets else pd.Series(dtype=float)
        diag = {
            "wd": wd_series,
            "V": v_series,
            "partitions": partitions_log,
            "n_positions": n_pos_log,
        }
        return s, diag
