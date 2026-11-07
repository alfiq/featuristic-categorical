# Categorical benchmark

`categorical_benchmark.py` compares feature synthesis on datasets where the signal
lives in categorical columns (group statistics, high-cardinality levels, category
crossings) plus two controls (pure numeric, and two real OpenML datasets).

## Protocol

- 75/25 train/test split per seed, up to 8,000 rows per dataset, seeds 0, 1, 2.
- Feature synthesis is fit on the training split only, with the same budget for
  every library: population 100, 20 generations, 8 features.
- The downstream model (RidgeCV for regression, LogisticRegression for
  classification) sees the baseline features (one-hot categoricals + scaled
  numerics) **plus** the synthesized features.
- Score: held-out R² (regression) or ROC AUC (`adult`, classification).

## Results

Mean over 3 seeds of the held-out score with baseline + synthesized features.
The worst seed is in parentheses where it differs a lot from the mean.

| Dataset | Baseline only | Upstream (Nim 2.0) | **This fork (Nim 2.0 + categorical stage)** | Previous fork (pure Python, islands) |
|---|---|---|---|---|
| group_deviation | -0.01 | 0.12 | **0.97** | 0.97 |
| cat_interaction | 0.20 | 0.38 | **0.97** | 0.97 |
| cat_level_200 | 0.98 | 0.98 | 0.98 | 0.98 |
| numeric_only | 0.47 | 0.88 | 0.94 (0.85) | 0.86 |
| adult (AUC) | 0.90 | 0.90 | 0.90 | 0.90 |
| diamonds | 0.94 | 0.79 (0.56) | **0.95** (0.87) | 0.95 |
| Fit time per run | – | 0.1–0.3 s | 0.2–1.1 s | 5–139 s |

How to read it:

- The categorical stage is what wins `group_deviation` and `cat_interaction`.
  With `categorical_interactions=False` this fork scores the same as upstream
  (0.14 and 0.36), see `results/hybrid_catint_off.json`.
- `cat_level_200` and `adult` are ties: one-hot / target encoding already
  captures the signal.
- `numeric_only` has no categorical columns, so the new stage does nothing there.
  The gap to upstream is run-to-run noise of the Nim GA.

## Caveats

- The Nim GA is not deterministic even with a fixed `random_state`: the same seed
  can give noticeably different scores between runs (seen on `numeric_only`).
- `diamonds` is unstable on the Nim engine with or without the categorical
  stage. With the stage off, one seed scored −14.5 (mean −4.3).
- Multiclass targets were not benchmarked.

## Files

`results/` holds the raw per-seed rows (baseline, synth-only and
baseline + synth scores, fit and transform times):

| File | What ran |
|---|---|
| `upstream_nim.json` | martineastwood/featuristic, `nim` branch |
| `hybrid_catint_on.json` | this fork, `categorical_interactions=True` (default) |
| `hybrid_catint_off.json` | this fork, `categorical_interactions=False` |
| `fork_python_islands.json` | previous pure-Python fork with independent GA islands |

## Running it

```bash
python benchmarks/categorical_benchmark.py out.json
```

Optional environment variables: `BENCH_ROWS`, `BENCH_POP`, `BENCH_GENS`,
`BENCH_SEEDS` (for example `0,1,2`), `BENCH_NFEAT`, `BENCH_JOBS`, and
`BENCH_CATINT` (`1`/`0`, to turn the categorical stage on or off). Pass dataset
names after the output file to run only some of them. The OpenML datasets are
downloaded on first use.
