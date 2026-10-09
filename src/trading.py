"""
Market-neutral contrarian trading at the cluster level.

At each rebalance date T:
  * Within each cluster, over the previous ``w = 5`` trading days, compute
    each stock's cumulative deviation from the cluster mean return. Stocks
    with sum > p (default 0) are previous winners (SHORT); sum < -p are
    previous losers (LONG).
  * Within each side, allocate equal weight; normalize long and short sides
    to equal gross dollar exposure.
  * Hold for up to ``l = 3`` trading days. Apply end-of-day stop-win at 2%
    cumulative portfolio return.
  * Aggregate active clusters with equal capital at the portfolio level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_portfolio_weights(
    cluster_tickers: list[str],
    ret_window: pd.DataFrame,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Return a dict ticker -> weight for one cluster-level contrarian book."""
    sub = ret_window[cluster_tickers]
    cluster_mean = sub.mean(axis=1)  # one value per day
    dev = sub.sub(cluster_mean, axis=0)
    dev_sum = dev.sum(axis=0)  # per-ticker sum of 5-day deviations
    winners = [t for t in cluster_tickers if dev_sum[t] > threshold]
    losers = [t for t in cluster_tickers if dev_sum[t] < -threshold]
    if not winners or not losers:
        return {}
    w = {}
    for t in winners:
        w[t] = -1.0 / len(winners)
    for t in losers:
        w[t] = 1.0 / len(losers)
    return w


def aggregate_cluster_weights(
    clusters: list[list[str]],
    ret_window: pd.DataFrame,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Equal-capital aggregation across active clusters."""
    books = []
    for cl in clusters:
        wb = cluster_portfolio_weights(cl, ret_window, threshold)
        if wb:
            books.append(wb)
    if not books:
        return {}
    agg: dict[str, float] = {}
    scale = 1.0 / len(books)
    for wb in books:
        for t, w in wb.items():
            agg[t] = agg.get(t, 0.0) + scale * w
    return agg


def simulate_trade_pnl(
    weights: dict[str, float],
    ret_future: pd.DataFrame,
    holding_period: int = 3,
    stop_win: float = 0.02,
) -> list[float]:
    """Run one trade: enter with ``weights``, hold up to ``holding_period``
    days, exit end-of-day when cumulative return >= stop_win.

    Returns a list of per-day realized returns (one entry per actual day held).
    """
    if not weights:
        return [0.0]
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    sub = ret_future[tickers].to_numpy()
    n_days = min(holding_period, sub.shape[0])
    cum = 0.0
    out: list[float] = []
    for d in range(n_days):
        day_ret = float(np.nansum(w * sub[d]))
        cum = (1.0 + cum) * (1.0 + day_ret) - 1.0
        out.append(day_ret)
        if cum >= stop_win:
            break
    return out


def run_backtest(
    ret: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    cluster_fn,
    corr_lookback: int = 5,
    holding_period: int = 3,
    stop_win: float = 0.02,
    threshold: float = 0.0,
) -> tuple[pd.Series, list[dict]]:
    """Generic cluster-strategy backtester.

    Parameters
    ----------
    ret : pd.DataFrame
        Daily returns (either raw or residual).
    rebalance_dates : list of Timestamp
        Dates at which to rebalance (entries at T use info through T-1).
    cluster_fn : callable
        Given rebalance date T, returns ``list[list[str]]`` (clusters of
        tickers). The function is responsible for its own correlation /
        clustering pipeline using info available at T.

    Returns
    -------
    (daily_return_series, log) : the realized daily-return Series over the
    active window and a list of per-rebalance diagnostic dicts.
    """
    idx = ret.index
    daily_rets: dict[pd.Timestamp, float] = {}
    log: list[dict] = []
    i = 0
    while i < len(rebalance_dates):
        T = rebalance_dates[i]
        T_pos = idx.get_indexer([T])[0]
        if T_pos < corr_lookback:
            i += 1
            continue
        if T_pos + holding_period > len(idx):
            break
        ret_window = ret.iloc[T_pos - corr_lookback: T_pos]
        clusters = cluster_fn(T)
        weights = aggregate_cluster_weights(clusters, ret_window, threshold)
        trade_rets = simulate_trade_pnl(
            weights, ret.iloc[T_pos: T_pos + holding_period],
            holding_period=holding_period, stop_win=stop_win,
        )
        for d, r in enumerate(trade_rets):
            daily_rets[idx[T_pos + d]] = r
        log.append({"date": T, "n_clusters": len(clusters), "n_positions": len(weights)})
        # Skip ahead by number of days actually held.
        held = len(trade_rets)
        # Advance to next rebalance date that is at least `held` days ahead.
        next_pos = T_pos + max(held, 1)
        # Find next rebalance_dates entry >= next_pos.
        while i < len(rebalance_dates) and idx.get_indexer([rebalance_dates[i]])[0] < next_pos:
            i += 1
        if i < len(rebalance_dates) and rebalance_dates[i] == T:
            i += 1
    if not daily_rets:
        return pd.Series(dtype=float), log
    s = pd.Series(daily_rets).sort_index()
    return s, log


def make_rebalance_calendar(
    index: pd.DatetimeIndex, start_pos: int, freq: int = 3
) -> list[pd.Timestamp]:
    """Return every ``freq``-th trading day starting at ``start_pos``."""
    return list(index[start_pos::freq])
