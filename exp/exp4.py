"""
Experiment 4: Robustness and scaling studies.

Part A -- Sample-budget scaling at fixed universe size (default N = 30):
    H = round(F * N * log N) with multipliers F in {1e-2, 1e-1, 1, 1e1, 1e2}
    on GBS Boost, GBS Roots and QIC-GBS, year 2020.

Part B -- Universe-size scaling at fixed sample budget (H = 5*N*log N):
    sweep N over the configured range (default 12..30 by step 3) on
    GBS Boost, GBS Roots, QIC-GBS and SPONGE for comparison.

Outputs:
  * results/exp4A_raw.csv, results/exp4A_summary.csv
  * results/exp4A_tr_vs_F.png
  * results/exp4A_wd_vs_F.png
  * results/exp4B_raw.csv, results/exp4B_summary.csv
  * results/exp4B_tr_vs_N.png
  * results/exp4B_wd_vs_N.png
"""

from __future__ import annotations

import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .common import (
    RESULTS_DIR, build_universe_returns, load_cfg, run_strategy,
)


def _plot_vs_x(
    df: pd.DataFrame, x: str, metric: str, out_path, title: str,
    log_x: bool = False,
) -> None:
    """Mean metric with +/- std band versus ``x`` for each method."""
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10")
    for i, (method, sub) in enumerate(df.groupby("method")):
        agg = sub.groupby(x)[metric].agg(["mean", "std"]).reset_index()
        ax.plot(agg[x], agg["mean"], marker="o", label=method, color=palette[i])
        ax.fill_between(
            agg[x], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
            alpha=0.15, color=palette[i],
        )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _resolve_budget(method: str, F: float, N: int) -> dict:
    """Translate a budget multiplier F into the strategy's H_factor kwargs."""
    H_target = max(1, round(F * N * math.log(max(N, 2))))
    # H_factor * N * log N = H_target  =>  H_factor = H_target / (N * log N)
    denom = max(1.0, N * math.log(max(N, 2)))
    H_factor_eff = H_target / denom
    kwargs: dict = {}
    if method == "gbs_boost":
        kwargs["H_factor"] = H_factor_eff
    elif method == "gbs_roots":
        kwargs["H_factor_roots"] = H_factor_eff
    elif method == "qic_gbs":
        kwargs["H_factor_qic"] = H_factor_eff
    return kwargs


def run_exp4A() -> pd.DataFrame:
    """Part A: sample-budget scaling."""
    cfg = load_cfg()
    e4 = cfg["exp4"]
    params = cfg["params"]
    year = int(e4["year"])
    n_stocks = int(e4["partA_N"])
    methods = list(e4["partA_methods"])
    F_grid = list(e4["F_grid"])
    n_seeds = int(e4["n_seeds"])

    print(f"[exp4A] Loading universe for {year}, N={n_stocks} ...")
    ret, mkt, _ = build_universe_returns(year=year, n_stocks=n_stocks)

    rows: list[dict] = []
    for method in methods:
        for F in F_grid:
            extra_base = _resolve_budget(method, F, n_stocks)
            for seed in range(n_seeds):
                _, _, metrics = run_strategy(
                    ret, mkt, method, year, seed, params, extra=extra_base,
                )
                rows.append({
                    "method": method, "F": F, "seed": seed, **metrics,
                })
                print(f"[exp4A] {method:10s} F={F:.3g} seed={seed} "
                      f"TR={metrics['TR']:+.4f} WD={metrics['WD']:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "exp4A_raw.csv", index=False)
    agg = raw.groupby(["method", "F"]).agg(
        TR_mean=("TR", "mean"), TR_std=("TR", "std"),
        Sharpe_mean=("Sharpe", "mean"), Sharpe_std=("Sharpe", "std"),
        WD_mean=("WD", "mean"), WD_std=("WD", "std"),
    ).reset_index()
    agg.to_csv(RESULTS_DIR / "exp4A_summary.csv", index=False)
    print("\n[exp4A] Summary:")
    print(agg.round(4).to_string(index=False))

    _plot_vs_x(raw, "F", "TR", RESULTS_DIR / "exp4A_tr_vs_F.png",
               "Exp 4A: TR vs GBS sample-budget multiplier F (N fixed)",
               log_x=True)
    _plot_vs_x(raw, "F", "WD", RESULTS_DIR / "exp4A_wd_vs_F.png",
               "Exp 4A: WD vs GBS sample-budget multiplier F", log_x=True)
    return raw


def run_exp4B() -> pd.DataFrame:
    """Part B: universe-size scaling at H = 5 * N * log N."""
    cfg = load_cfg()
    e4 = cfg["exp4"]
    params = cfg["params"]
    year = int(e4["year"])
    methods = list(e4["partB_methods"])
    N_grid = list(e4["partB_N_grid"])
    n_seeds = int(e4["n_seeds"])
    F_fixed = float(e4.get("partB_F", 5.0))

    rows: list[dict] = []
    for N in N_grid:
        print(f"[exp4B] === N = {N} ===")
        ret, mkt, _ = build_universe_returns(year=year, n_stocks=N)
        for method in methods:
            extra_base = _resolve_budget(method, F_fixed, N)
            for seed in range(n_seeds):
                _, _, metrics = run_strategy(
                    ret, mkt, method, year, seed, params, extra=extra_base,
                )
                rows.append({
                    "method": method, "N": N, "seed": seed, **metrics,
                })
                print(f"[exp4B] N={N} {method:10s} seed={seed} "
                      f"TR={metrics['TR']:+.4f} WD={metrics['WD']:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "exp4B_raw.csv", index=False)
    agg = raw.groupby(["method", "N"]).agg(
        TR_mean=("TR", "mean"), TR_std=("TR", "std"),
        Sharpe_mean=("Sharpe", "mean"), Sharpe_std=("Sharpe", "std"),
        WD_mean=("WD", "mean"), WD_std=("WD", "std"),
    ).reset_index()
    agg.to_csv(RESULTS_DIR / "exp4B_summary.csv", index=False)
    print("\n[exp4B] Summary:")
    print(agg.round(4).to_string(index=False))

    _plot_vs_x(raw, "N", "TR", RESULTS_DIR / "exp4B_tr_vs_N.png",
               "Exp 4B: TR vs universe size N (H = 5 N log N)")
    _plot_vs_x(raw, "N", "WD", RESULTS_DIR / "exp4B_wd_vs_N.png",
               "Exp 4B: WD vs universe size N")
    return raw


def run_exp4() -> dict:
    """Run both parts of Experiment 4."""
    a = run_exp4A()
    b = run_exp4B()
    return {"partA": a, "partB": b}


if __name__ == "__main__":
    run_exp4()
