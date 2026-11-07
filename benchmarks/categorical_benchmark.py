"""Benchmark categorical handling: this fork (in-GA categorical operators) vs
upstream featuristic 2.x (encode categoricals up front, then evolve numerics).

Run the same script in two environments (one with each library installed):

    python benchmarks/categorical_benchmark.py out_mine.json
    python benchmarks/categorical_benchmark.py out_upstream.json

Protocol per dataset and seed:
  * 75/25 train/test split.
  * Fit GeneticFeatureSynthesis on train only (same GA budget for both libs).
  * Downstream model sees: baseline features (one-hot cats + scaled numerics,
    identical for both libs) + the synthesized features.
  * Report test score (R^2 or ROC AUC) and synthesis fit time.
"""

import json
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import featuristic as ft

warnings.filterwarnings("ignore")

IS_UPSTREAM = hasattr(ft, "featuristic_lib")
# BENCH_CATINT=0/1 toggles the hybrid categorical stage on the Nim version
# (ignored by builds that do not have the option).
import inspect as _inspect

HAS_CATINT = (
    IS_UPSTREAM
    and "categorical_interactions"
    in _inspect.signature(ft.GeneticFeatureSynthesis.__init__).parameters
)
_CATINT_ENV = __import__("os").environ.get("BENCH_CATINT")
if IS_UPSTREAM:
    LIB = "upstream"
    if HAS_CATINT and _CATINT_ENV is not None:
        LIB = "hybrid" if _CATINT_ENV == "1" else "hybrid_off"
else:
    LIB = "mine"
# Knobs (env vars) so the run can be scaled up gradually from a tiny smoke test.
import os

POP = int(os.environ.get("BENCH_POP", "100"))
GENS = int(os.environ.get("BENCH_GENS", "20"))
NFEAT = int(os.environ.get("BENCH_NFEAT", "8"))
SEEDS = [int(s) for s in os.environ.get("BENCH_SEEDS", "0,1,2").split(",")]
MAX_ROWS = int(os.environ.get("BENCH_ROWS", "8000"))  # cap per dataset
# The fork's ParallelPopulation starts one process per worker, each holding a
# copy of the data; n_jobs=-1 on every core is what exhausted RAM before.
JOBS = int(os.environ.get("BENCH_JOBS", "1"))


# ----------------------------------------------------------------- datasets
def ds_group_deviation(seed=0, n=4000):
    """y depends on x minus its group mean -> needs a group aggregation."""
    r = np.random.default_rng(seed)
    city = r.choice([f"c{i}" for i in range(30)], n)
    offs = {c: r.normal(0, 5) for c in np.unique(city)}
    price = np.array([offs[c] for c in city]) + r.normal(0, 1, n)
    gm = pd.Series(price).groupby(city).transform("mean").values
    y = np.abs(price - gm) * 3 + r.normal(0, 0.3, n)  # nonlinear in the deviation
    X = pd.DataFrame({"city": city, "price": price, "noise": r.normal(size=n)})
    return X, pd.Series(y), "reg"


def ds_cat_level(seed=0, n=4000):
    """y = per-category level + linear numeric term -> target encoding wins."""
    r = np.random.default_rng(seed)
    cat = r.choice([f"k{i}" for i in range(200)], n)
    lv = {c: r.normal(0, 3) for c in np.unique(cat)}
    x = r.normal(size=n)
    y = np.array([lv[c] for c in cat]) + 2 * x + r.normal(0, 0.5, n)
    return pd.DataFrame({"cat": cat, "x": x}), pd.Series(y), "reg"


def ds_cat_interaction(seed=0, n=4000):
    """y driven by the *pair* (a, b), not by a or b alone -> needs crossing."""
    r = np.random.default_rng(seed)
    a = r.choice(list("ABCDEFGH"), n)
    b = r.choice(list("pqrstuvw"), n)
    pair = {(i, j): r.normal(0, 3) for i in "ABCDEFGH" for j in "pqrstuvw"}
    y = np.array([pair[(i, j)] for i, j in zip(a, b)]) + r.normal(0, 0.5, n)
    X = pd.DataFrame({"a": a, "b": b, "x": r.normal(size=n)})
    return X, pd.Series(y), "reg"


def ds_numeric_only(seed=0, n=4000):
    """No categoricals: pure symbolic regression (sanity / speed check)."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = X.x0 * X.x1 + np.sin(X.x2) * 3 + X.x3**2 + r.normal(0, 0.3, n)
    return X, y, "reg"


def ds_adult(seed=0):
    from sklearn.datasets import fetch_openml

    d = fetch_openml("adult", version=2, as_frame=True)
    X = d.data.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(str)
    X = X.fillna("missing")  # string cols only; numeric cols have no NaN here
    y = (d.target.astype(str).str.contains(">50K")).astype(int)
    idx = X.sample(8000, random_state=seed).index
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True), "clf"


def ds_diamonds(seed=0):
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=42225, as_frame=True)  # diamonds
    X = d.data.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(str)
    y = np.log(d.target.astype(float))
    idx = X.sample(8000, random_state=seed).index
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True), "reg"


DATASETS = {
    "group_deviation": ds_group_deviation,
    "cat_level_200": ds_cat_level,
    "cat_interaction": ds_cat_interaction,
    "numeric_only": ds_numeric_only,
    "adult (openml)": ds_adult,
    "diamonds (openml)": ds_diamonds,
}


# ----------------------------------------------------------------- helpers
def baseline_tf(X):
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X.columns if c not in cats]
    return ColumnTransformer(
        [
            (
                "c",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=5, sparse_output=False
                ),
                cats,
            ),
            ("n", StandardScaler(), nums),
        ]
    )


def make_synth(seed):
    common = {
        "population_size": POP,
        "max_generations": GENS,
        "return_all_features": False,
        "pbar": False,
        "verbose": False,
    }
    if IS_UPSTREAM:
        if HAS_CATINT and _CATINT_ENV is not None:
            common["categorical_interactions"] = _CATINT_ENV == "1"
        return ft.GeneticFeatureSynthesis(n_features=NFEAT, random_state=seed, **common)
    random.seed(seed)
    np.random.seed(seed)
    return ft.GeneticFeatureSynthesis(
        num_features=NFEAT, n_jobs=JOBS, random_state=seed, **common
    )


def numeric_block(F):
    F = pd.DataFrame(F).apply(pd.to_numeric, errors="coerce")
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return F.values


def score(task, Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = np.clip(sc.transform(Xtr), -10, 10), np.clip(sc.transform(Xte), -10, 10)
    if task == "reg":
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, ytr)
        return r2_score(yte, m.predict(Xte))
    m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def run_one(name, fn, seed):
    X, y, task = fn(seed)
    if len(X) > MAX_ROWS:
        idx = X.sample(MAX_ROWS, random_state=seed).index
        X, y = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    ytr, yte = ytr.reset_index(drop=True), yte.reset_index(drop=True)

    bt = baseline_tf(Xtr).fit(Xtr)
    Btr, Bte = bt.transform(Xtr), bt.transform(Xte)
    base = score(task, Btr, ytr, Bte, yte)

    synth = make_synth(seed)
    t0 = time.perf_counter()
    synth.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ftr, Fte = numeric_block(synth.transform(Xtr)), numeric_block(synth.transform(Xte))
    tr_s = time.perf_counter() - t0

    only = score(task, Ftr, ytr, Fte, yte)
    plus = score(task, np.hstack([Btr, Ftr]), ytr, np.hstack([Bte, Fte]), yte)
    return {
        "dataset": name,
        "seed": seed,
        "task": task,
        "lib": LIB,
        "baseline": base,
        "synth_only": only,
        "baseline_plus_synth": plus,
        "fit_s": fit_s,
        "transform_s": tr_s,
    }


def main(out):
    rows = []
    only = sys.argv[2:] or list(DATASETS)
    for name in only:
        for seed in SEEDS:
            try:
                r = run_one(name, DATASETS[name], seed)
            except Exception as e:  # noqa: BLE001 - keep going; record the failure
                r = {
                    "dataset": name,
                    "seed": seed,
                    "lib": LIB,
                    "error": f"{type(e).__name__}: {e}"[:300],
                }
            print(json.dumps(r), flush=True)
            rows.append(r)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
