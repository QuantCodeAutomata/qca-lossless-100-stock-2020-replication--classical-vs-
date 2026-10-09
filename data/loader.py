"""
Daily equity data loader using the Massive REST API.

Responsibilities:
  * Download daily OHLCV bars via `RESTClient.list_aggs`.
  * Download cash dividends via `RESTClient.list_dividends`.
  * Download Fama-French daily market excess return via pandas-datareader.
  * Compute dividend-aware daily total returns:
        R_{t,i} = (P_{t,i} + D_{t,i}) / P_{t-1,i} - 1
    where D_{t,i} is the per-share cash dividend with ex_dividend_date == t.
  * Cache per-ticker raw bars and dividends to local Parquet files (gitignored).

The cache directory lives under `data/cache/` and is excluded from version
control via `.gitignore`. Only the deliverable artifacts under `results/`
are committed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Massive client
# ---------------------------------------------------------------------------
def _client():
    # Context7 confirmed: official `massive` package `RESTClient` is the
    # supported REST entry point.
    from massive import RESTClient

    api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("MASSIVE_TOKEN")
    if not api_key:
        raise RuntimeError(
            "MASSIVE_API_KEY (or MASSIVE_TOKEN) environment variable is required."
        )
    return RESTClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Daily bars
# ---------------------------------------------------------------------------
def _bars_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"bars_{ticker}.pkl"


def _dividends_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"divs_{ticker}.pkl"


def _read_cache(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_pickle(path)


def load_daily_bars(
    ticker: str, start: str, end: str, use_cache: bool = True
) -> pd.DataFrame:
    """Return a DataFrame of daily unadjusted OHLCV for `ticker`.

    Columns: ``open, high, low, close, volume, vwap, transactions`` indexed by
    tz-aware UTC date (converted to tz-naive date for alignment).
    """
    cache = _bars_cache_path(ticker)
    if use_cache and cache.exists():
        df = _read_cache(cache)
        if not df.empty:
            dfr = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            # If cache fully covers the window, return it.
            if dfr.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=5) \
                    and dfr.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7):
                return dfr

    client = _client()
    rows: list[dict] = []
    for a in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start,
        to=end,
        limit=50000,
    ):
        # Timestamp is Unix milliseconds for daily aggregate bars.
        rows.append(
            {
                "timestamp": a.timestamp,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
                "vwap": a.vwap,
                "transactions": a.transactions,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(
        "America/New_York"
    ).dt.normalize().dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume", "vwap", "transactions"]]
    _write_cache(df, cache)
    return df


def load_dividends(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """Return dividend history for `ticker`: columns ``ex_date, cash_amount``."""
    cache = _dividends_cache_path(ticker)
    if use_cache and cache.exists():
        return _read_cache(cache)

    client = _client()
    rows: list[dict] = []
    try:
        for d in client.list_dividends(ticker=ticker, limit=1000):
            if d.ex_dividend_date is None or d.cash_amount is None:
                continue
            rows.append(
                {"ex_date": pd.Timestamp(d.ex_dividend_date), "cash_amount": float(d.cash_amount)}
            )
    except Exception:
        rows = []
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ex_date").drop_duplicates("ex_date")
    _write_cache(df, cache)
    return df


def load_panel(
    tickers: Iterable[str], start: str, end: str, pause: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download bars+dividends for a list of tickers.

    Returns ``(close_df, dividend_df)`` wide DataFrames indexed by date with
    columns = tickers. Missing days remain NaN (no forward fill).
    """
    close_cols: dict[str, pd.Series] = {}
    div_cols: dict[str, pd.Series] = {}
    for t in tickers:
        bars = load_daily_bars(t, start, end)
        if bars.empty:
            continue
        close_cols[t] = bars["close"]
        divs = load_dividends(t)
        if not divs.empty:
            d = divs.set_index("ex_date")["cash_amount"]
            d.index = d.index.normalize()
            div_cols[t] = d
        if pause:
            time.sleep(pause)
    if not close_cols:
        raise RuntimeError("No data downloaded for any ticker in the universe.")
    close_df = pd.concat(close_cols, axis=1).sort_index()
    close_df.columns.name = "ticker"

    # Build dividend matrix aligned on the trading calendar.
    div_df = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    for t, s in div_cols.items():
        s_in_cal = s.reindex(close_df.index).fillna(0.0)
        div_df[t] = s_in_cal
    return close_df, div_df


# ---------------------------------------------------------------------------
# Fama-French factors (daily)
# ---------------------------------------------------------------------------
def load_ff_market_excess(start: str, end: str) -> pd.Series:
    """Daily Fama-French market excess return R_mkt - R_f (as decimal)."""
    cache = CACHE_DIR / "ff_daily.pkl"
    if cache.exists():
        s = _read_cache(cache).iloc[:, 0]
    else:
        from pandas_datareader import data as pdr

        ff = pdr.DataReader(
            "F-F_Research_Data_Factors_daily", "famafrench",
            start="1990-01-01", end="2025-12-31",
        )[0]
        if hasattr(ff.index, "to_timestamp"):
            ff.index = ff.index.to_timestamp()
        ff.index = pd.to_datetime(ff.index).normalize()
        # Fama-French returns are in percent.
        s = ff["Mkt-RF"].astype(float) / 100.0
        s.name = "mkt_excess"
        _write_cache(s.to_frame(), cache)
    return s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# Daily total returns R_{t,i} = (P_t + D_t) / P_{t-1} - 1
# ---------------------------------------------------------------------------
def total_returns(close_df: pd.DataFrame, div_df: pd.DataFrame) -> pd.DataFrame:
    """Dividend-aware daily total returns as defined in the paper."""
    assert close_df.shape == div_df.shape, "price/dividend matrices must align"
    # Compute (P_t + D_t) / P_{t-1} - 1
    numer = close_df + div_df.reindex_like(close_df).fillna(0.0)
    denom = close_df.shift(1)
    ret = numer / denom - 1.0
    # Do NOT forward-fill. Drop first row (NaN from shift).
    ret = ret.iloc[1:]
    return ret


# ---------------------------------------------------------------------------
# Convenience: full load for a universe
# ---------------------------------------------------------------------------
def load_universe_returns(
    tickers: Iterable[str], start: str, end: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (ret_df, mkt_excess_series) aligned on trading days.

    Only trading days with both FF market data and at least one stock price
    are kept. Columns are filtered to tickers with complete data over the
    requested window.
    """
    close_df, div_df = load_panel(tickers, start, end)
    ret = total_returns(close_df, div_df)
    mkt = load_ff_market_excess(start, end)
    common = ret.index.intersection(mkt.index)
    ret = ret.loc[common]
    mkt = mkt.loc[common]
    # Drop tickers with any NaN inside the window (complete-data policy).
    good = ret.columns[~ret.isna().any(axis=0)]
    ret = ret[good]
    return ret, mkt
