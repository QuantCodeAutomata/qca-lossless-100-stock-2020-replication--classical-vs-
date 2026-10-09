# GBS Clustering for Statistical Arbitrage — Replication Repository

Python replication of the four experiments described in the task spec:

| id   | title                                                                 |
|------|-----------------------------------------------------------------------|
| exp1 | Lossless 100-stock 2020: classical vs GBS-based clustering            |
| exp2 | 50-stock 2020 photon-loss & displacement study                        |
| exp3 | Multi-regime no-loss comparison (2008, 2017, 2020, 2022, 50 stocks)   |
| exp4 | Robustness: sample-budget scaling and universe-size scaling           |

All experiments share the same residual-return correlation graph pipeline:

1. Dividend-aware daily total return `R = (P_t + D_t)/P_{t-1} - 1`
2. 60-day rolling market-β neutralisation against Fama–French `Mkt-RF`
3. 5-day residual correlation matrix at each rebalance
4. Market-mode removal and Marchenko–Pastur noise filtering (`ρ=N/T`,
   `λ_+ = (1+√ρ)²`)
5. Clustering: Spectral, symmetric SPONGE, GBS Boost, GBS Roots, QIC-GBS,
   random baseline
6. Contrarian market-neutral book inside each cluster: short previous
   winners, long previous losers; hold ≤ 3 days with 2 % stop-win

GBS sampling uses Strawberry Fields with `n̄ = √N_current` recomputed after
every residual-graph reduction. Exp 2 adds a channel-loss model `η=1-l_r`
and an optional coherent-displacement compensation surrogate.

## Repository layout

```
src/     core library (one module per concern)
exp/     one script per experiment (python -m exp.exp1 ...)
tests/   pytest suite (unit tests, no live Massive calls)
data/    Massive REST loader + SP500 universe builder; cache is gitignored
config/  config.yaml: year, universe size, methods, seeds, grids
results/ metrics CSVs, plots and RESULTS.md (committed deliverable)
```

## Running

```bash
pip install -r requirements.txt
export MASSIVE_TOKEN=$MASSIVE_API_KEY     # loader reads MASSIVE_TOKEN
pytest -q                                  # 27 tests, no network
python -m exp.exp1                         # ~1.5 min at N=40, 2 seeds
python -m exp.exp2                         # ~0.7 min at N=25
python -m exp.exp3                         # ~2.5 min, 4 years
python -m exp.exp4                         # ~6 min, F- and N-sweeps
```

All runtime parameters (universe size, seed count, methods, grids) live in
`config/config.yaml`. The paper-matching values (N=100 for exp1, N=50 for
exp2/3) are the configuration defaults the authors used; the repo defaults
to compute-friendly sizes for a reproducible end-to-end run on a single
machine. The documented pipeline is independent of universe size.

## Data

`data/loader.py` wraps `massive.RESTClient`. Daily bars are pulled with
`list_aggs(..., timespan="day")` and cached on disk under `data/cache/`.
The cache is gitignored — only `results/` outputs are committed.

Dividend events are pulled from `client.list_dividends()` when available
and merged in on their `pay_date`. The Fama–French daily `Mkt-RF` factor
is fetched through `pandas_datareader` (Ken French library).

## Deliverables

See [`results/RESULTS.md`](results/RESULTS.md) for the full write-up of
all four experiments with metric tables and figures.
