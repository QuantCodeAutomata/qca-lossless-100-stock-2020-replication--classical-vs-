"""
Experiment 2: Loss and displacement study on a 50-stock 2020 universe.

For a grid of photon-loss rates l_r in [0, 0.9]:
  * Run GBS Roots with no displacement (standard lossy GBS sampler).
  * Run GBS Roots with displacement compensation (``displace=True``).
  * Run a random-clustering baseline matched to the compensated-GBS
    cluster-size profile at zero loss.

Metrics per (method, loss, seed): TR, Sharpe, Sortino, WD, V.

Outputs:
  * results/exp2_raw.csv
  * results/exp2_summary.csv
  * results/exp2_tr_vs_loss.png
  * results/exp2_wd_vs_loss.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .common import (
    RESULTS_DIR, build_universe_returns, load_cfg, run_strategy,
)


def _plot_curve(df: pd.DataFrame, metric: str, out_path, title: str):
    """Loss-response curve with uncertainty band for one metric."""
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10")
    for i, (cond, sub) in enumerate(df.groupby("condition")):
        agg = sub.groupby("loss")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(agg["loss"], agg["mean"], marker="o", label=cond, color=palette[i])
        ax.fill_between(
            agg["loss"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
            alpha=0.15, color=palette[i],
        )
    ax.set_xlabel("Photon loss rate")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.axhline(0.0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_exp2() -> pd.DataFrame:
    """Execute Experiment 2 and persist all outputs."""
    cfg = load_cfg()
    e2 = cfg["exp2"]
    params = cfg["params"]
    year = int(e2["year"])
    n_stocks = int(e2["n_stocks"])
    methods = list(e2["methods"])
    loss_grid = list(e2["loss_grid"])
    n_seeds = int(e2["n_seeds"])
    include_random = bool(e2.get("include_random_baseline", True))

    print(f"[exp2] Loading universe for {year}, N={n_stocks} ...")
    ret, mkt, _uni = build_universe_returns(year=year, n_stocks=n_stocks)

    rows: list[dict] = []
    # ``no_disp`` and ``displaced`` conditions for each GBS method.
    for method in methods:
        for loss in loss_grid:
            for displace in (False, True):
                condition = f"{method}_{'displaced' if displace else 'nodisp'}"
                for seed in range(n_seeds):
                    extra = {"loss": loss, "displace": displace}
                    series, diag, metrics = run_strategy(
                        ret, mkt, method, year, seed, params, extra=extra,
                    )
                    rows.append({
                        "method": method,
                        "condition": condition,
                        "loss": loss,
                        "seed": seed,
                        **metrics,
                    })
                    print(f"[exp2] {condition:22s} loss={loss:.2f} seed={seed} "
                          f"TR={metrics['TR']:+.4f} ShR={metrics['Sharpe']:+.3f} "
                          f"WD={metrics['WD']:.3f}")

    # Random baseline uses cluster-size distribution from the zero-loss
    # compensated GBS runs; the baseline is loss-independent in construction
    # but is logged at every loss level for comparability on plots.
    if include_random:
        for loss in loss_grid:
            for seed in range(n_seeds):
                extra = {
                    "loss": 0.0,  # random baseline unaffected by GBS loss
                    "random_sizes": None,
                }
                series, diag, metrics = run_strategy(
                    ret, mkt, "random", year, seed, params, extra=extra,
                )
                rows.append({
                    "method": "random",
                    "condition": "random_baseline",
                    "loss": loss,
                    "seed": seed,
                    **metrics,
                })
                print(f"[exp2] random_baseline      loss={loss:.2f} seed={seed} "
                      f"TR={metrics['TR']:+.4f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "exp2_raw.csv", index=False)

    agg = raw.groupby(["condition", "loss"]).agg(
        TR_mean=("TR", "mean"), TR_std=("TR", "std"),
        Sharpe_mean=("Sharpe", "mean"), Sharpe_std=("Sharpe", "std"),
        Sortino_mean=("Sortino", "mean"), Sortino_std=("Sortino", "std"),
        WD_mean=("WD", "mean"), WD_std=("WD", "std"),
        V_mean=("V", "mean"), V_std=("V", "std"),
    ).reset_index()
    agg.to_csv(RESULTS_DIR / "exp2_summary.csv", index=False)
    print("\n[exp2] Summary:")
    print(agg.round(4).to_string(index=False))

    _plot_curve(raw, "TR", RESULTS_DIR / "exp2_tr_vs_loss.png",
                "Exp 2: total return vs photon loss (2020, 50-stock)")
    _plot_curve(raw, "WD", RESULTS_DIR / "exp2_wd_vs_loss.png",
                "Exp 2: weighted density vs photon loss")

    return raw


if __name__ == "__main__":
    run_exp2()
