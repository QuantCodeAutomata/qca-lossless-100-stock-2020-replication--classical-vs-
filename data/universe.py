"""
Universe selection rule.

Deterministic proxy for the paper's stratified selection procedure:
  * Rank eligible stocks by prior-calendar-year total return including dividends.
  * Allocate sector counts proportional to the full eligible pool using largest
    remainder rounding so sum of quotas equals N exactly.
  * Break ties by ticker string (deterministic, since no reliable live market
    cap feed is available via the Massive REST API for historical point-in-time
    date).
"""

from __future__ import annotations

import pandas as pd

from .loader import load_universe_returns
from .sp500_universe import pool_df


def _largest_remainder(sector_counts: dict[str, int], N: int) -> dict[str, int]:
    """Allocate N slots across sectors proportionally with largest remainder."""
    total = sum(sector_counts.values())
    quotas = {s: N * c / total for s, c in sector_counts.items()}
    floors = {s: int(q) for s, q in quotas.items()}
    allocated = sum(floors.values())
    remainder = N - allocated
    # Rank sectors by fractional remainder descending; break ties by sector name.
    frac = sorted(
        quotas.items(),
        key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]),
    )
    for i in range(remainder):
        floors[frac[i][0]] += 1
    return floors


def select_universe(
    year: int, n_stocks: int, pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select ``n_stocks`` for test year ``year``.

    Ranking is based on prior calendar year total return (including dividends),
    with sector-proportional quotas derived from the eligible pool.

    Returns a DataFrame with columns ``ticker, sector, prev_year_return``.
    """
    if pool is None:
        pool = pool_df()
    prev_start = f"{year - 1}-01-01"
    prev_end = f"{year - 1}-12-31"

    # Download prior-year returns for the entire pool.
    tickers = pool["ticker"].tolist()
    ret, _mkt = load_universe_returns(tickers, prev_start, prev_end)
    # Eligible pool = tickers with complete prior-year data.
    eligible = pool[pool["ticker"].isin(ret.columns)].copy()
    prev_tr = (1.0 + ret).prod(axis=0) - 1.0
    eligible["prev_year_return"] = eligible["ticker"].map(prev_tr)
    eligible = eligible.dropna(subset=["prev_year_return"])

    # Build sector proportional quotas on the eligible pool.
    sector_counts = eligible["sector"].value_counts().to_dict()
    quotas = _largest_remainder(sector_counts, min(n_stocks, len(eligible)))

    # Within each sector, pick top-ranked tickers.
    picks: list[pd.DataFrame] = []
    for sector, q in quotas.items():
        if q <= 0:
            continue
        sub = eligible[eligible["sector"] == sector].sort_values(
            ["prev_year_return", "ticker"], ascending=[False, True]
        ).head(q)
        picks.append(sub)
    out = pd.concat(picks, ignore_index=True).sort_values(
        ["sector", "ticker"]
    ).reset_index(drop=True)
    return out
