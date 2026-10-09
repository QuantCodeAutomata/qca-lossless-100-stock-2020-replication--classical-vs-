"""
Shared helpers for all experiment scripts.

* Load the YAML configuration.
* Build the per-year 50/100-stock universe via the data loader.
* Run a single ``ClusterStrategy`` configuration and compute summary metrics.
* Aggregate multiple seeds into mean +/- std tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.loader import load_universe_returns
from data.universe import select_universe
from src.metrics import summarize_run
from src.pipeline import ClusterStrategy

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_cfg() -> dict:
    """Load the YAML config file."""
    with open(ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def build_universe_returns(
    year: int, n_stocks: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return ``(ret, mkt, universe_df)`` for a given test year.

    The window used for data loading starts one year before the test year to
    satisfy the 60-day beta warm-up and includes the full test year.
    """
    uni = select_universe(year=year, n_stocks=n_stocks)
    start = f"{year - 1}-01-01"
    end = f"{year}-12-31"
    tickers = uni["ticker"].tolist()
    ret, mkt = load_universe_returns(tickers, start, end)
    # Drop tickers that lost data in the merge.
    uni = uni[uni["ticker"].isin(ret.columns)].reset_index(drop=True)
    ret = ret[uni["ticker"].tolist()]
    return ret, mkt, uni


def run_strategy(
    ret: pd.DataFrame,
    mkt: pd.Series,
    method: str,
    year: int,
    seed: int,
    params: dict,
    extra: dict | None = None,
) -> tuple[pd.Series, dict, dict[str, float]]:
    """Instantiate and run a single strategy configuration.

    Returns the daily return series, the diagnostic dict, and a metric
    dictionary suitable for aggregation.
    """
    extra = extra or {}
    strat = ClusterStrategy(
        ret=ret,
        mkt=mkt,
        method=method,
        beta_window=params["beta_window"],
        corr_lookback=params["corr_lookback"],
        holding_period=params["holding_period"],
        stop_win=params["stop_win"],
        rebalance_freq=params["holding_period"],
        threshold_p=params["deviation_threshold"],
        k_B=params["perm_B"],
        k_alpha=params["mp_alpha"],
        seed=seed,
        start_date=pd.Timestamp(f"{year}-01-01"),
        end_date=pd.Timestamp(f"{year}-12-31"),
        **extra,
    )
    series, diag = strat.run()
    metrics = summarize_run(
        series, diag["wd"], diag["V"], diag["partitions"],
        rf_daily=params.get("rf_rate_daily", 0.0),
    )
    return series, diag, metrics


def aggregate_seeds(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean and std of numeric fields across seeds."""
    df = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for col in df.columns:
        if df[col].dtype.kind in "iufb":
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std(ddof=0))
    return out


def welch_t_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Two-sided Welch's t-test for difference in means."""
    from scipy import stats

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"t_stat": float("nan"), "p_value": float("nan")}
    t_stat, p_value = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}
