"""Tests for the hybrid Python categorical-interaction stage."""

import pickle

import numpy as np
import pandas as pd
import pytest

import featuristic as ft
from featuristic.synthesis.categorical_features import CategoricalFeatureGenerator


def _group_data(n=600, seed=0):
    r = np.random.default_rng(seed)
    city = r.choice([f"c{i}" for i in range(10)], n)
    offs = {c: r.normal(0, 5) for c in np.unique(city)}
    price = np.array([offs[c] for c in city]) + r.normal(0, 1, n)
    gm = pd.Series(price).groupby(city).transform("mean").to_numpy()
    y = np.abs(price - gm) * 3 + r.normal(0, 0.3, n)
    X = pd.DataFrame({"city": city, "price": price, "noise": r.normal(size=n)})
    return X, pd.Series(y)


def test_group_stats_frozen_and_unseen_fallback():
    X, y = _group_data()
    gen = CategoricalFeatureGenerator(max_features=None)
    tr = gen.fit_transform(X, y)
    assert "city__gdiff__price" in tr.columns
    means = X.groupby("city")["price"].mean()
    np.testing.assert_allclose(
        tr["city__gdiff__price"], X["price"] - X["city"].map(means), atol=1e-9
    )

    Xt = pd.DataFrame(
        {"city": ["c0", "never_seen"], "price": [1.0, 2.0], "noise": [0, 0]}
    )
    out = gen.transform(Xt)
    assert np.isfinite(out.to_numpy()).all()
    # Unseen category -> average training group value; frequency -> 0.
    assert out.loc[1, "city__gmean__price"] == pytest.approx(means.mean())
    assert out.loc[0, "city__gmean__price"] == pytest.approx(means["c0"])
    assert out.loc[1, "city__freq"] == 0.0
    # Transform of new data must not change with the batch it comes in.
    single = gen.transform(Xt.iloc[[1]])
    np.testing.assert_allclose(single.to_numpy(), out.iloc[[1]].to_numpy())


def test_pair_target_encoding_does_not_leak():
    # Pure-noise target with many tiny pair cells: an in-sample encoding would
    # correlate strongly with y; a cross-fitted one must not.
    r = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame(
        {
            "a": r.choice([f"a{i}" for i in range(40)], n),
            "b": r.choice([f"b{i}" for i in range(40)], n),
        }
    )
    y = pd.Series(r.normal(size=n))
    gen = CategoricalFeatureGenerator(max_features=None, random_state=0)
    tr = gen.fit_transform(X, y)
    col = "a__x__b__te"
    assert col in tr.columns
    leak_corr = abs(np.corrcoef(tr[col], y)[0, 1])
    assert leak_corr < 0.1, leak_corr


def test_cap_and_continuous_columns_not_grouped():
    r = np.random.default_rng(2)
    n = 500
    X = pd.DataFrame({f"c{i}": r.choice(list("abcd"), n) for i in range(6)})
    for i in range(6):
        X[f"x{i}"] = r.normal(size=n)  # continuous -> never a group key
    y = pd.Series(r.normal(size=n))
    gen = CategoricalFeatureGenerator(max_features=20)
    tr = gen.fit_transform(X, y)
    assert tr.shape[1] == 20
    assert set(gen.cat_cols_) == {f"c{i}" for i in range(6)}
    assert not any(c.startswith("x") for c in tr.columns)


def test_gfs_option_off_matches_upstream_and_on_transform_consistent():
    X, y = _group_data(400)
    kw = {
        "n_features": 3,
        "population_size": 20,
        "max_generations": 3,
        "pbar": False,
        "random_state": 0,
        "return_all_features": True,
    }
    off = ft.GeneticFeatureSynthesis(categorical_interactions=False, **kw).fit(X, y)
    assert off.categorical_generator_ is None
    assert not any("__" in str(c) for c in off.feature_names_)
    assert off.feature_names_ == ["city", "price", "noise"]

    on = ft.GeneticFeatureSynthesis(**kw).fit(X, y)
    assert on.categorical_generator_ is not None
    assert "city__gdiff__price" in on.feature_names_

    Xt, _ = _group_data(50, seed=9)
    a = on.transform(Xt)
    b = pickle.loads(pickle.dumps(on)).transform(Xt)
    pd.testing.assert_frame_equal(a, b)
    # Row-wise: transforming a subset gives the same rows.
    c = on.transform(Xt.iloc[:10])
    np.testing.assert_allclose(c.to_numpy(), a.iloc[:10].to_numpy())
    assert np.isfinite(a.to_numpy(dtype=float)).all()


def test_gfs_numeric_only_unchanged_by_option():
    r = np.random.default_rng(3)
    X = pd.DataFrame(r.normal(size=(300, 3)), columns=list("abc"))
    y = X.a * X.b + r.normal(0, 0.1, 300)
    kw = {
        "n_features": 2,
        "population_size": 20,
        "max_generations": 3,
        "pbar": False,
        "random_state": 0,
    }
    on = ft.GeneticFeatureSynthesis(**kw).fit(X, y)
    off = ft.GeneticFeatureSynthesis(categorical_interactions=False, **kw).fit(X, y)
    assert on.categorical_generator_ is None
    pd.testing.assert_frame_equal(on.transform(X), off.transform(X))
