import featuristic as ft
import pytest
import pandas as pd


def test_gfs():
    num_features = 5
    gfs = ft.GeneticFeatureSynthesis(num_features=num_features, pbar=False)

    with pytest.raises(Exception):
        gfs.plot_history()

    assert gfs.len_hall_of_fame == num_features * 5
    assert gfs.fit_called == False

    with pytest.raises(Exception):
        gfs.fit(X=None, y=None)

    # X needs enough variance/rows for simple correlations to not be trivial/singular
    X = pd.DataFrame({"a": [1, 2, 4, 5, 8, 10, 12, 14, 15, 20], "b": [4, 5, 6, 2, 1, 8, 9, 3, 4, 10]})
    y = pd.Series([1, 2, 3, 5, 8, 9, 10, 3, 4, 8])

    gfs.fit(X, y)
    new_X = gfs.transform(X)
    new_cols = [x for x in new_X.columns if x.startswith("feature_")]
    assert len(new_cols) == num_features
    assert gfs.fit_called == True

    gfs = ft.GeneticFeatureSynthesis(num_features=num_features, pbar=False)
    new_X = gfs.fit_transform(X, y)
    new_cols = [x for x in new_X.columns if x.startswith("feature_")]
    assert len(new_cols) == num_features
