import featuristic as ft
import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np


def objective_function(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return scores.mean() * -1


def test_selection():
    np.random.seed(8888)
    gfs = ft.GeneticFeatureSelector(objective_function)

    with pytest.raises(Exception):
        gfs.fit(X=None, y=None)

    X = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [10, 10, 10], "d": [1, 1, 1]}
    )
    y = pd.Series([1, 2, 3])

    gfs.fit(X, y)
    new_X = gfs.transform(X)
    selected_cols = new_X.columns.tolist()
    # With complexity penalty, it should only pick 'a' or 'b' (not both, as they are redundant)
    # and definitely not 'c' or 'd' which are constants.
    assert len(selected_cols) > 0
    assert set(selected_cols).issubset({"a", "b"})
    assert "c" not in selected_cols
    assert "d" not in selected_cols
    assert new_X.shape[0] == 3
    assert gfs.is_fitted_ == True
