"""
Experiment 3: Market-regime comparison without loss on 50-stock universes.

Years: {2008, 2017, 2020, 2022}.  Each year uses a fresh 50-stock universe
built from prior-year top performers with sector-proportional quotas.

Methods: Spectral, SPONGE, GBS Boost, GBS Roots, QIC-GBS.

A static equal-weighted 1/N buy-and-hold benchmark is also computed on the
same 50-stock universe for each year, over the same active trading window.

Outputs:
  * results/exp3_raw.csv
  * results/exp3_summary.csv
  * results/exp3_benchmark.csv
  * results/exp3_tr_by_regime.png
  * results/exp3_sharpe_by_regime.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .common import (
    RESULTS_DIR, aggregate_seeds, build_universe_returns, load_cfg,
    run_strategy,
)


def _buy_and_hold_return(ret: pd.DataFrame, year: int) -> dict[str, float]:
    """1/N equal-weighted buy-and-hold return for the year."""
    sub = ret[(ret.index.year == year)].dropna(how="all")
    if sub.empty:
        return {"BH_TR": 0.0, "BH_Sharpe": 0.0}
    eq = sub.mean(axis=1)
    cum = (1.0 + eq).prod() - 1.0
    sd = eq.std(ddof=0)
    sharpe = float(eq.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0
    return {"BH_TR": float(cum), "BH_Sharpe": sharpe}


def run_exp3() -> pd.DataFrame:
    """Run Experiment 3 across all four regime years."""
    cfg = load_cfg()
    e3 = cfg["exp3"]
    params = cfg["params"]
    years = list(e3["years"])
    n_stocks = int(e3["n_stocks"])
    methods = list(e3["methods"])
    n_seeds = int(e3["n_seeds"])

    rows: list[dict] = []
    bh_rows: list[dict] = []
    for year in years:
        print(f"\n[exp3] === Year {year} ===")
        ret, mkt, _ = build_universe_returns(year=year, n_stocks=n_stocks)
        bh = _buy_and_hold_return(ret, year)
        bh_rows.append({"year": year, **bh})
        print(f"[exp3] 1/N buy-and-hold {year}: TR={bh['BH_TR']:+.4f}, "
              f"Sharpe={bh['BH_Sharpe']:+.3f}")
        for method in methods:
            for seed in range(n_seeds):
                series, diag, metrics = run_strategy(
                    ret, mkt, method, year, seed, params,
                )
                rows.append({"year": year, "method": method, "seed": seed, **metrics})
                print(f"[exp3] {year} {method:12s} seed={seed} "
                      f"TR={metrics['TR']:+.4f} ShR={metrics['Sharpe']:+.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "exp3_raw.csv", index=False)
    bh_df = pd.DataFrame(bh_rows)
    bh_df.to_csv(RESULTS_DIR / "exp3_benchmark.csv", index=False)

    agg_rows: list[dict] = []
    for (year, method), sub in raw.groupby(["year", "method"]):
        agg = aggregate_seeds(sub.drop(columns=["year", "method", "seed"])
                              .to_dict("records"))
        agg_rows.append({"year": year, "method": method, **agg})
    summary = pd.DataFrame(agg_rows)
    summary.to_csv(RESULTS_DIR / "exp3_summary.csv", index=False)
    print("\n[exp3] Summary:")
    print(summary.round(3).to_string(index=False))

    # Bar charts: TR and Sharpe by year/method, with 1/N benchmark overlay.
    for metric, bh_col, title, fname in [
        ("TR_mean", "BH_TR",
         "Exp 3: TR by regime (bars) with 1/N benchmark (red line)",
         "exp3_tr_by_regime.png"),
        ("Sharpe_mean", "BH_Sharpe",
         "Exp 3: Sharpe by regime with 1/N benchmark (red line)",
         "exp3_sharpe_by_regime.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.barplot(
            data=summary, x="year", y=metric, hue="method", ax=ax,
            palette="tab10",
        )
        for _, row in bh_df.iterrows():
            x = years.index(row["year"])
            ax.hlines(row[bh_col], x - 0.4, x + 0.4,
                      colors="red", linewidth=2.5, linestyles="--")
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_mean", ""))
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / fname, dpi=140)
        plt.close(fig)
    return raw


if __name__ == "__main__":
    run_exp3()
