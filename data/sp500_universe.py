"""
S&P 500 constituent proxy list with sector classification.

A full point-in-time CRSP S&P 500 membership snapshot is not available in the
Massive API. For reproducibility we use a fixed hand-curated subset of large,
long-listed S&P 500 constituents with their GICS sector classification. The
subset covers all 11 GICS sectors used in the paper. The implementation of the
universe-selection rule (previous-year total-return ranking with sector
proportional quotas) operates on this fixed pool.
"""

from __future__ import annotations

# Fixed pool of long-listed S&P 500 constituents with GICS sector.
# This is used as the eligible universe for the stratified selection rule.
SP500_POOL: list[tuple[str, str]] = [
    # Energy
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"), ("SLB", "Energy"),
    ("EOG", "Energy"), ("PSX", "Energy"), ("MPC", "Energy"), ("OXY", "Energy"),
    ("VLO", "Energy"), ("HAL", "Energy"),
    # Materials
    ("LIN", "Materials"), ("APD", "Materials"), ("SHW", "Materials"),
    ("NEM", "Materials"), ("FCX", "Materials"), ("DOW", "Materials"),
    ("NUE", "Materials"), ("DD", "Materials"),
    # Industrials
    ("GE", "Industrials"), ("HON", "Industrials"), ("UPS", "Industrials"),
    ("CAT", "Industrials"), ("BA", "Industrials"), ("MMM", "Industrials"),
    ("LMT", "Industrials"), ("DE", "Industrials"), ("RTX", "Industrials"),
    ("UNP", "Industrials"), ("FDX", "Industrials"), ("CSX", "Industrials"),
    ("NSC", "Industrials"), ("EMR", "Industrials"), ("ETN", "Industrials"),
    # Consumer Discretionary
    ("AMZN", "Consumer Discretionary"), ("HD", "Consumer Discretionary"),
    ("MCD", "Consumer Discretionary"), ("NKE", "Consumer Discretionary"),
    ("LOW", "Consumer Discretionary"), ("SBUX", "Consumer Discretionary"),
    ("TJX", "Consumer Discretionary"), ("F", "Consumer Discretionary"),
    ("GM", "Consumer Discretionary"), ("TGT", "Consumer Discretionary"),
    ("BKNG", "Consumer Discretionary"), ("MAR", "Consumer Discretionary"),
    # Consumer Staples
    ("PG", "Consumer Staples"), ("KO", "Consumer Staples"),
    ("PEP", "Consumer Staples"), ("WMT", "Consumer Staples"),
    ("COST", "Consumer Staples"), ("MDLZ", "Consumer Staples"),
    ("CL", "Consumer Staples"), ("KMB", "Consumer Staples"),
    ("GIS", "Consumer Staples"), ("MO", "Consumer Staples"),
    ("PM", "Consumer Staples"), ("KHC", "Consumer Staples"),
    # Healthcare
    ("JNJ", "Healthcare"), ("UNH", "Healthcare"), ("PFE", "Healthcare"),
    ("ABT", "Healthcare"), ("MRK", "Healthcare"), ("TMO", "Healthcare"),
    ("LLY", "Healthcare"), ("ABBV", "Healthcare"), ("MDT", "Healthcare"),
    ("BMY", "Healthcare"), ("AMGN", "Healthcare"), ("DHR", "Healthcare"),
    ("GILD", "Healthcare"), ("CI", "Healthcare"),
    # Financials
    ("JPM", "Financials"), ("BAC", "Financials"), ("WFC", "Financials"),
    ("C", "Financials"), ("GS", "Financials"), ("MS", "Financials"),
    ("AXP", "Financials"), ("BLK", "Financials"), ("USB", "Financials"),
    ("PNC", "Financials"), ("SCHW", "Financials"), ("CB", "Financials"),
    ("MET", "Financials"), ("AIG", "Financials"), ("TRV", "Financials"),
    # Information Technology
    ("AAPL", "Information Technology"), ("MSFT", "Information Technology"),
    ("NVDA", "Information Technology"), ("INTC", "Information Technology"),
    ("CSCO", "Information Technology"), ("ORCL", "Information Technology"),
    ("IBM", "Information Technology"), ("ADBE", "Information Technology"),
    ("TXN", "Information Technology"), ("QCOM", "Information Technology"),
    ("ACN", "Information Technology"), ("AMD", "Information Technology"),
    ("HPQ", "Information Technology"), ("MU", "Information Technology"),
    ("AMAT", "Information Technology"),
    # Communication Services
    ("GOOGL", "Communication Services"), ("META", "Communication Services"),
    ("DIS", "Communication Services"), ("NFLX", "Communication Services"),
    ("CMCSA", "Communication Services"), ("VZ", "Communication Services"),
    ("T", "Communication Services"), ("CHTR", "Communication Services"),
    # Utilities
    ("NEE", "Utilities"), ("DUK", "Utilities"), ("SO", "Utilities"),
    ("D", "Utilities"), ("AEP", "Utilities"), ("EXC", "Utilities"),
    ("SRE", "Utilities"), ("XEL", "Utilities"),
    # Real Estate
    ("AMT", "Real Estate"), ("PLD", "Real Estate"), ("EQIX", "Real Estate"),
    ("CCI", "Real Estate"), ("PSA", "Real Estate"), ("SPG", "Real Estate"),
    ("O", "Real Estate"), ("WELL", "Real Estate"),
]


def pool_df():
    """Return the SP500 pool as a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame(SP500_POOL, columns=["ticker", "sector"])
