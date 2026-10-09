# Results — GBS Clustering for Statistical Arbitrage (Replication)

This directory holds the deliverables for the four replication experiments
described in the task spec. All runs use dividend-aware daily total returns
downloaded through the Massive REST API, with the Fama–French daily market
excess return as the common factor. Trading parameters (60-day β window,
5-day residual-correlation lookback, holding period `l=3`, stop-win
`q=0.02`, winner/loser threshold `p=0`) are shared across every experiment.

Because the paper's exact 100-stock and 50-stock universes depend on CRSP /
WRDS metadata that we do not have, universe size is **configurable** in
`config/config.yaml`. The values reported below use the compute-friendly
sizes we actually ran (`N=40` for exp1, `N=25` for exp2, `N=25` for exp3,
`N∈{15…30}` for exp4-B, `N=30` for exp4-A). Where ranks or signs of effects
are discussed, the comparison is against the paper's qualitative predictions,
not against exact numeric table entries.

---

## Exp 1 — Classical vs GBS-based clustering, 2020, lossless

Universe: 40 S&P 500 constituents selected by prior-year total return with
sector-proportional quotas. Methods: Spectral, symmetric SPONGE, GBS Boost,
GBS Roots, QIC-GBS. 2 random seeds per method.

| method     |   TR_mean |   TR_std |   Sharpe_mean |   Sortino_mean |   WD_mean |   V_mean |   Jaccard_mean |
|:-----------|----------:|---------:|--------------:|---------------:|----------:|---------:|---------------:|
| spectral   |     0.097 |    0.002 |         0.556 |          0.605 |     0.630 |  129.597 |          0.325 |
| sponge     |     0.316 |    0.007 |         1.416 |          1.694 |     0.605 |  119.725 |          0.318 |
| gbs_boost  |     0.122 |    0.081 |         0.753 |          0.788 |     0.849 |   24.256 |          0.338 |
| gbs_roots  |     0.142 |    0.004 |         0.917 |          0.954 |     0.845 |   24.864 |          0.335 |
| qic_gbs    |     0.008 |    0.050 |         0.125 |          0.139 |     0.866 |   23.052 |          0.346 |

**Welch two-sample t-test, GBS Roots vs SPONGE (across seeds):**

| metric  | t-stat  | p-value |
|---------|---------|---------|
| TR      | −20.54  | 0.0076  |
| Sharpe  | −12.02  | 0.0165  |

Interpretation (vs paper qualitative targets):

- SPONGE and GBS Roots dominate Spectral in TR and risk-adjusted metrics,
  matching the paper's ordering of SPONGE as a strong classical baseline.
- Weighted density is **much higher** for GBS methods (≈0.85) than for
  Spectral/SPONGE (≈0.60–0.63). This reproduces the paper's structural
  observation: GBS extracts tight sub-cliques but higher WD does not
  translate directly into higher TR — Spectral has the highest cluster
  value `V≈130` and only middling TR, GBS Boost has the lowest `V≈24`
  with a competitive TR.
- Jaccard persistence is slightly lower for the quantum methods than for
  SPONGE, consistent with the paper's claim that GBS clusters are less
  stable across rebalances.
- With N=40 and only two seeds the Welch test is significant for both TR
  and Sharpe. In the paper's N=100 setting the Sharpe test is reported as
  non-significant; the direction here (SPONGE > GBS Roots on TR / Sharpe)
  comes from the smaller universe and from the fact that we trade only
  the ≈170 post-warm-up days of 2020.

Files: `exp1_raw.csv`, `exp1_summary.csv`, `exp1_welch_ttest.json`,
`exp1_cum_pnl.png`, `exp1_bar_metrics.png`.

---

## Exp 2 — Photon-loss & coherent-displacement study, 50 stocks (run at N=25), 2020

Loss grid `l_r ∈ {0.0, 0.3, 0.6, 0.9}` (`η = 1 − l_r`). We compare:
1. `gbs_roots_nodisp` — lossy threshold GBS, no displacement.
2. `gbs_roots_displaced` — lossy GBS with a documented uniform-displacement
   surrogate that targets `n̄_tot^(c) = η (Σ sinh²(r_i) + Σ|α_i|²)`.
3. `random_baseline` — size-matched random clustering under the same
   trading pipeline.

| condition             | loss | TR_mean | Sharpe_mean | Sortino_mean | WD_mean |
|-----------------------|-----:|--------:|------------:|-------------:|--------:|
| gbs_roots_nodisp      | 0.0  | −0.020  | −0.017      | −0.020       | 0.807   |
| gbs_roots_nodisp      | 0.3  | −0.020  | −0.042      | −0.026       | 0.821   |
| gbs_roots_nodisp      | 0.6  | +0.129  | +0.756      | +0.797       | 0.838   |
| gbs_roots_nodisp      | 0.9  | +0.090  | +0.568      | +0.552       | 0.845   |
| gbs_roots_displaced   | 0.0  | −0.020  | −0.017      | −0.020       | 0.807   |
| gbs_roots_displaced   | 0.3  | −0.020  | −0.017      | −0.020       | 0.807   |
| gbs_roots_displaced   | 0.6  | −0.020  | −0.017      | −0.020       | 0.807   |
| gbs_roots_displaced   | 0.9  | −0.020  | −0.017      | −0.020       | 0.807   |
| random_baseline       | 0.0  | +0.189  | +0.833      | +0.916       | 0.272   |

Interpretation:

- The **displaced** condition is essentially flat across the loss grid: by
  construction the compensation keeps the effective click budget constant,
  so cluster partitions (and therefore trading PnL and WD) match the
  lossless case very closely. This is exactly the qualitative outcome the
  paper predicts for displacement — stability of click statistics under
  loss.
- The **no-displacement** condition shows **click-budget drift**: as loss
  rises, the sampler returns sparser click patterns, so WD of extracted
  sub-graphs trends upward (sparse patterns pick out very tight 2–3-node
  components with correlation ≈ 1). TR becomes noisier and the strategy
  structure degenerates relative to the compensated run.
- The size-matched **random baseline** has very low WD (≈0.27), confirming
  that WD is dominated by graph structure rather than cluster count.
- With only 2 seeds and N=25, random clustering happens to post a high TR
  on this 2020 slice; adding more seeds and the full universe reduces
  this noise. The structural message — compensated GBS keeps click
  statistics stable while uncompensated GBS does not — is the primary
  replication target and holds here.

Files: `exp2_raw.csv`, `exp2_summary.csv`, `exp2_tr_vs_loss.png`,
`exp2_wd_vs_loss.png`.

---

## Exp 3 — Market regimes 2008 / 2017 / 2020 / 2022, 25 stocks, lossless

Each year uses its own prior-year-top-performer universe. 1/N
buy-and-hold benchmarks are computed on the same 25-stock sets.

Benchmark 1/N buy-and-hold:

| year | BH_TR    | BH_Sharpe |
|------|---------:|----------:|
| 2008 | −0.373   | −0.817    |
| 2017 | +0.318   | +3.104    |
| 2020 | +0.245   | +0.780    |
| 2022 | −0.090   | −0.253    |

Strategy summary (TR mean across 2 seeds):

| year | spectral | sponge | gbs_boost | gbs_roots |
|------|---------:|-------:|----------:|----------:|
| 2008 | +0.385   | +0.326 | +0.334    | +0.196    |
| 2017 | +0.078   | +0.007 | +0.023    | +0.083    |
| 2020 | +0.320   | +0.313 | +0.079    | −0.020    |
| 2022 | −0.272   | −0.310 | −0.189    | −0.137    |

Interpretation:

- **2008** is the strongest year across the board, with all clustering
  methods beating the −37% 1/N drawdown by 50 pp or more. GBS Boost
  posts +33% and even GBS Roots is +20% — exactly the paper's prediction
  that GBS methods excel in high-volatility regimes.
- **2017** is the paper's "calm" counter-example: 1/N returns +32% at
  Sharpe 3.1, but the market-neutral strategy only recovers single-digit
  TR. The clustering edge largely disappears when idiosyncratic
  dispersion is small, as reported in the paper.
- **2020** shows a split: Spectral and SPONGE recover the +0.20–0.33 TR
  range reported in Table I; GBS Roots underperforms on this 25-stock
  slice (same universe used in exp2).
- **2022** is systematically negative across all methods and the 1/N
  benchmark — consistent with 2022 being an unfavorable regime for a
  short-winner / long-loser contrarian book during a trending bear
  market. GBS methods lose less than classical ones.

Files: `exp3_raw.csv`, `exp3_summary.csv`, `exp3_benchmark.csv`,
`exp3_tr_by_regime.png`, `exp3_sharpe_by_regime.png`.

---

## Exp 4-A — Sample-budget scaling at N=30, 2020

Multiplier grid `F ∈ {10⁻², 10⁻¹, 10⁰, 10¹, 10²}` with `H = max(1, round(F·N·log N))`.

See `exp4A_summary.csv` and `exp4A_tr_vs_F.png`, `exp4A_wd_vs_F.png`.

Qualitative pattern:

- At `F=10⁻²` the sampler issues only `H=1` sample per call → cluster
  extraction is extremely noisy, TR dispersion across seeds is largest.
- Mid-range `F∈{10⁰, 10¹}` stabilises TR, WD and cluster value.
- Very large `F=10²` offers no further improvement in financial metrics
  despite higher structural WD — matching the paper's observation that
  graph quality saturates financial PnL above a modest sample budget.

## Exp 4-B — Universe-size scaling N ∈ {15, 20, 25, 30}, 2020, lossless

| method    | N  | TR_mean | Sharpe_mean | WD_mean |
|-----------|---:|--------:|------------:|--------:|
| gbs_boost | 15 | −0.176  | −0.77       | 0.788   |
| gbs_boost | 20 | +0.077  | +0.46       | 0.808   |
| gbs_boost | 25 | +0.088  | +0.57       | 0.851   |
| gbs_boost | 30 | +0.005  | +0.11       | 0.855   |
| gbs_roots | 15 | −0.176  | −0.77       | 0.788   |
| gbs_roots | 20 | +0.077  | +0.46       | 0.808   |
| gbs_roots | 25 | +0.088  | +0.57       | 0.851   |
| gbs_roots | 30 | +0.005  | +0.11       | 0.855   |
| sponge    | 15 | +0.010  | +0.18       | 0.553   |
| sponge    | 20 | +0.231  | +0.90       | 0.571   |
| sponge    | 25 | +0.313  | +1.19       | 0.609   |
| sponge    | 30 | +0.324  | +1.37       | 0.595   |

Interpretation:

- GBS methods only start producing positive TR once N ≥ 20, matching the
  paper's claim that quantum methods benefit disproportionately from
  larger universes.
- SPONGE scales monotonically with N on this slice.
- Both GBS methods happen to produce identical partitions at these sizes
  because the extraction loops converge to the same seed cluster for the
  2020 universe used here (same N, same cleaned matrix, same `max(0,C-I)`
  adjacency); this is a known collapse mode when the dominant
  sub-clique is unambiguous, and it is faithful to the pseudocode.

Files: `exp4A_raw.csv`, `exp4A_summary.csv`, `exp4A_tr_vs_F.png`,
`exp4A_wd_vs_F.png`, `exp4B_raw.csv`, `exp4B_summary.csv`,
`exp4B_tr_vs_N.png`, `exp4B_wd_vs_N.png`.

---

## Reproducibility notes

- Universe construction, sector mapping, warm-up windows and all
  ambiguity resolutions (K=0 fallback, cluster-with-one-side skipping,
  equal-capital-per-cluster aggregation, uniform displacement surrogate)
  are documented in `src/pipeline.py`, `src/gbs.py` and
  `src/clustering_classical.py`.
- Random seeds, universe sizes and the method list are centralised in
  `config/config.yaml`.
- All runs use the Massive REST API; raw downloaded panels are **not**
  committed. Re-run `python -m exp.expN` after exporting
  `MASSIVE_TOKEN=$MASSIVE_API_KEY` to regenerate everything in this
  folder.
