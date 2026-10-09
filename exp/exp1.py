"""
Experiment 1: Lossless replication - Classical vs GBS-based clustering.

Target year: 2020 on a (reduced-size proxy of) 100-stock S&P 500 universe.
Methods: Spectral, SPONGE (symmetric), GBS Boost, GBS Roots, QIC-GBS.
Per-method metrics are aggregated over multiple random seeds and a Welch
two-sample t-test is run between GBS Roots and SPONGE on terminal returns
and Sharpe ratios.

Outputs:
  * results/exp1_summary.csv            -- per method mean/std of metrics
  * results/exp1_raw.csv                -- per method,seed metrics row
  * results/exp1_cum_pnl.png            -- cumulative-return trajectories
  * results/exp1_bar_metrics.png        -- bar chart TR / Sharpe / Sortino
  * results/exp1_welch_ttest.json       -- t-test details
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .common import (
    RESULTS_DIR, aggregate_seeds, build_universe_returns, load_cfg,
    run_strategy, welch_t_test,
)


def _plot_cum_pnl(series_by_method: dict[str, list[pd.Series]], out_path):
    """Plot mean cumulative return per method with +/- std uncertainty band."""
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(series_by_method))
    for color, (method, series_list) in zip(palette, series_by_method.items()):
        if not series_list:
            continue
        all_idx = sorted(set().union(*[s.index for s in series_list]))
        aligned = pd.DataFrame({
            i: s.reindex(all_idx).fillna(0.0).values for i, s in enumerate(series_list)
        }, index=all_idx)
        cum = (1.0 + aligned).cumprod() - 1.0
        mean = cum.mean(axis=1)
        std = cum.std(axis=1)
        ax.plot(mean.index, mean.values, label=method, color=color, linewidth=1.8)
        ax.fill_between(
            mean.index, (mean - std).values, (mean + std).values,
            color=color, alpha=0.15,
        )
    ax.set_title("Exp 1: 2020 lossless cumulative return by clustering method")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_bar_metrics(summary: pd.DataFrame, out_path):
    """Grouped bar chart of mean TR / Sharpe / Sortino with std error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    colors = sns.color_palette("tab10", n_colors=len(summary.index))
    for ax, metric in zip(axes, ["TR_mean", "Sharpe_mean", "Sortino_mean"]):
        std_col = metric.replace("_mean", "_std")
        ax.bar(
            summary.index, summary[metric].values,
            yerr=summary[std_col].values, color=colors, capsize=4,
        )
        ax.set_title(metric.replace("_mean", ""))
        ax.tick_params(axis="x", rotation=25)
        ax.axhline(0.0, color="black", linewidth=0.5)
    fig.suptitle("Exp 1: 2020 lossless performance (mean +/- std across seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_exp1() -> dict:
    """Execute Experiment 1 end-to-end and persist all deliverables."""
    cfg = load_cfg()
    e1 = cfg["exp1"]
    params = cfg["params"]
    year = int(e1["year"])
    n_stocks = int(e1["n_stocks"])
    methods = list(e1["methods"])
    n_seeds = int(e1["n_seeds"])

    print(f"[exp1] Loading universe for year {year}, n_stocks={n_stocks} ...")
    ret, mkt, uni = build_universe_returns(year=year, n_stocks=n_stocks)
    print(f"[exp1] loaded ret shape={ret.shape}, "
          f"{uni['sector'].value_counts().to_dict()}")

    raw_rows: list[dict] = []
    series_by_method: dict[str, list[pd.Series]] = {m: [] for m in methods}
    for method in methods:
        for seed in range(n_seeds):
            series, diag, metrics = run_strategy(
                ret, mkt, method, year, seed, params,
            )
            series_by_method[method].append(series)
            raw_rows.append({"method": method, "seed": seed, **metrics})
            print(f"[exp1] {method:12s} seed={seed} "
                  f"TR={metrics['TR']:+.4f} ShR={metrics['Sharpe']:+.3f} "
                  f"SoR={metrics['Sortino']:+.3f} WD={metrics['WD']:.3f} "
                  f"V={metrics['V']:.3f} J={metrics['Jaccard']:.3f}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(RESULTS_DIR / "exp1_raw.csv", index=False)

    # Aggregate.
    summary_rows: list[dict] = []
    for method in methods:
        agg = aggregate_seeds(raw_df[raw_df["method"] == method]
                              .drop(columns=["method", "seed"]).to_dict("records"))
        summary_rows.append({"method": method, **agg})
    summary = pd.DataFrame(summary_rows).set_index("method")
    summary.to_csv(RESULTS_DIR / "exp1_summary.csv")
    print("\n[exp1] Summary:")
    print(summary.round(4).to_string())

    # Welch t-test: GBS Roots vs SPONGE on TR and Sharpe.
    tests: dict[str, dict] = {}
    if "gbs_roots" in methods and "sponge" in methods:
        a_tr = raw_df.loc[raw_df["method"] == "gbs_roots", "TR"].tolist()
        b_tr = raw_df.loc[raw_df["method"] == "sponge", "TR"].tolist()
        a_sh = raw_df.loc[raw_df["method"] == "gbs_roots", "Sharpe"].tolist()
        b_sh = raw_df.loc[raw_df["method"] == "sponge", "Sharpe"].tolist()
        tests["TR_gbs_roots_vs_sponge"] = welch_t_test(a_tr, b_tr)
        tests["Sharpe_gbs_roots_vs_sponge"] = welch_t_test(a_sh, b_sh)
        with open(RESULTS_DIR / "exp1_welch_ttest.json", "w") as fh:
            json.dump(tests, fh, indent=2)
        print(f"[exp1] Welch t-test TR     gbs_roots vs sponge: {tests['TR_gbs_roots_vs_sponge']}")
        print(f"[exp1] Welch t-test Sharpe gbs_roots vs sponge: {tests['Sharpe_gbs_roots_vs_sponge']}")

    _plot_cum_pnl(series_by_method, RESULTS_DIR / "exp1_cum_pnl.png")
    _plot_bar_metrics(summary, RESULTS_DIR / "exp1_bar_metrics.png")

    return {"summary": summary, "raw": raw_df, "tests": tests}


if __name__ == "__main__":
    run_exp1()
