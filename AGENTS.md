# AGENTS — repository notes

## What this repo is

Replication of a quantum-GBS-clustering-for-statistical-arbitrage paper
across four experiments (lossless 100-stock 2020, lossy 50-stock 2020,
multi-regime 2008/2017/2020/2022, and sample-budget / universe-size
scaling studies).

## Key gotchas

- **Massive API key**: the official `massive` client reads `MASSIVE_TOKEN`.
  Export it from `MASSIVE_API_KEY` before running experiments:
  `export MASSIVE_TOKEN=$MASSIVE_API_KEY`.
- **Universe size is configurable** in `config/config.yaml`. Paper
  targets (N=100, N=50) are kept as commented defaults; the committed
  defaults are compute-friendly sizes that produced the figures in
  `results/`.
- **Data cache**: `data/cache/` holds pickled Massive bars/dividends. It
  is gitignored. Do **not** commit it.
- **Seaborn barplot `yerr`**: `sns.barplot` in current seaborn versions
  no longer accepts `yerr`. `exp/exp1.py` uses `matplotlib.pyplot.bar`
  directly with `yerr=` for the error-bar chart. If you add new plots,
  prefer `ax.bar(..., yerr=...)`.
- **GBS collapse mode**: at small N the GBS Boost and GBS Roots
  extraction loops can converge to the same first-cluster seed and
  therefore produce identical partitions. This is faithful to the
  pseudocode in the paper and is noted in `results/RESULTS.md`.
- **Strawberry Fields API**: we use `sf.apps.sample.sample(A, n_mean,
  n_samples, threshold=True)`. For the lossy run in exp2 the loss
  channel is applied by rescaling the adjacency weights by `η = 1 - l_r`
  before sampling (standard small-squeezing approximation). Displacement
  compensation is a documented surrogate (uniform `α_i`) — swap-in point
  is `src/gbs.py::_apply_displacement`.

## Testing

`pytest -q` runs the full suite (27 tests) without any network calls.
All tests mock the loader or use synthetic matrices. CI-safe.

## Where to look when debugging

- Clustering outputs look wrong → `src/clustering_classical.py` (Spectral,
  SPONGE) or `src/clustering_gbs.py` (Boost, Roots, QIC).
- Trading PnL looks off → `src/trading.py` and `src/pipeline.py`
  (`run_backtest`). The 60/5/3/0.02 numbers live in `src/config.py`.
- RMT filtering → `src/residuals.py::clean_correlation`.
- GBS sampling under loss / displacement → `src/gbs.py::sample_gbs`.
