"""
Beta estimation and residual-return computation.

At each rebalance date T:
    * beta_i = Cov(R_i, R_mkt) / Var(R_mkt) over the trailing ``beta_window``
      trading days ending at T - 1.
    * residual return  R^{res}_{t,i} = R_{t,i} - beta_i * R_{mkt,t}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_betas(
    ret: pd.DataFrame, mkt: pd.Series, rebalance_date, window: int = 60
) -> pd.Series:
    """Return a Series of betas (indexed by ticker) estimated over the
    `window` trading days strictly before `rebalance_date`."""
    idx = ret.index
    end_pos = idx.get_indexer([pd.Timestamp(rebalance_date)])[0]
    if end_pos == -1:
        # Nearest previous date.
        end_pos = idx.searchsorted(pd.Timestamp(rebalance_date)) - 1
    start_pos = end_pos - window
    if start_pos < 0:
        raise ValueError(
            f"Not enough history before {rebalance_date} for a {window}-day beta."
        )
    r_win = ret.iloc[start_pos:end_pos]
    m_win = mkt.iloc[start_pos:end_pos]
    assert len(r_win) == window
    m = m_win.to_numpy()
    m_dev = m - m.mean()
    var_m = float((m_dev * m_dev).sum())
    if var_m <= 0:
        return pd.Series(0.0, index=ret.columns)
    betas = {}
    for c in r_win.columns:
        x = r_win[c].to_numpy()
        x_dev = x - x.mean()
        betas[c] = float((x_dev * m_dev).sum() / var_m)
    return pd.Series(betas)


def residual_returns(
    ret: pd.DataFrame, mkt: pd.Series, betas: pd.Series
) -> pd.DataFrame:
    """Compute residual returns R^{res}_{t,i} = R_{t,i} - beta_i * R_{mkt,t}."""
    aligned = mkt.reindex(ret.index)
    res = ret.sub(aligned.values[:, None] * betas.reindex(ret.columns).values[None, :], axis=0)
    return res
