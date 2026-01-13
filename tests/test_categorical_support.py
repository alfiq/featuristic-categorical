import featuristic as ft
import numpy as np
import pandas as pd
from pytest import approx

def test_symbolic_frequency_encoding():
    func = ft.synthesis.symbolic_functions.SymbolicFrequencyEncoding()
    s = pd.Series(['a', 'a', 'b', 'c'])
    # counts: a=2, b=1, c=1. totals=4. freqs: a=0.5, b=0.25, c=0.25
    expected = pd.Series([0.5, 0.5, 0.25, 0.25])
    pd.testing.assert_series_equal(func(s), expected)

def test_symbolic_length():
    func = ft.synthesis.symbolic_functions.SymbolicLength()
    s = pd.Series(['cat', 'doggy', 'a'])
    expected = pd.Series([3, 5, 1])
    pd.testing.assert_series_equal(func(s), expected.astype(int), check_dtype=False)

def test_symbolic_equals():
    func = ft.synthesis.symbolic_functions.SymbolicEquals()
    a = pd.Series(['cat', 'dog', 'cat'])
    b = pd.Series(['cat', 'cat', 'bird'])
    expected = pd.Series([1, 0, 0])
    pd.testing.assert_series_equal(func(a, b), expected.astype(int), check_dtype=False)

def test_robustness_on_strings():
    # Mathematical functions should not crash on strings, they should return NaNs
    add = ft.synthesis.symbolic_functions.SymbolicAdd()
    sin = ft.synthesis.symbolic_functions.SymbolicSin()
    mult = ft.synthesis.symbolic_functions.SymbolicMultiply()
    
    s = pd.Series(['a', 'b'])
    
    # Add string to string might actually work in pandas/numpy depending on ufunc, 
    # but sin("a") definitely fails.
    # Our implementation returns a Series of NaNs on Exception.
    
    res_sin = sin(s)
    # Result might be Series or Numpy array (optimization)
    if isinstance(res_sin, pd.Series):
        assert res_sin.isna().all()
    else:
        assert np.isnan(res_sin).all()
    
    assert len(res_sin) == 2

    # Test mult typo fix
    assert mult.format_str == "({} * {})"

def test_mrmr_with_categorical():
    from featuristic.synthesis.mrmr import MaxRelevanceMinRedundancy
    X = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['cat', 'dog', 'cat', 'bird', 'dog'], # non-numeric
        'C': [5, 4, 3, 2, 1]
    })
    y = pd.Series([10, 20, 15, 25, 30])
    
    selector = MaxRelevanceMinRedundancy(k=2)
    # This should not crash even if 'B' is non-numeric because it should be filtered out
    selector.fit(X, y)
    assert 'B' not in selector.selected_features
    assert 'A' in selector.selected_features or 'C' in selector.selected_features

def test_genetic_synthesis_integration():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'A': rng.rand(50),
        'B': ['cat', 'dog'] * 25,
        'target': rng.rand(50)
    })
    
    synth = ft.GeneticFeatureSynthesis(
        num_features=2,
        population_size=10,
        max_generations=2,
        n_jobs=1
    )
    # This integration test verifies that the whole pipeline works with mixed types
    synth.fit(df[['A', 'B']], df['target'])
    res = synth.transform(df[['A', 'B']])
    assert res.shape[1] > 2

def test_reproducibility():
    # Test CatCodes reproducibility
    func_cat = ft.synthesis.symbolic_functions.SymbolicCatCodes()
    train_data = pd.Series(['Apple', 'Banana', 'Apple'])
    test_data = pd.Series(['Banana'])
    
    res_train = func_cat(train_data)
    # Apple is 0, Banana is 1
    banana_code = res_train.iloc[1]
    
    res_test = func_cat(test_data)
    assert res_test.iloc[0] == banana_code
    
    # Test Frequency reproducibility
    func_freq = ft.synthesis.symbolic_functions.SymbolicFrequencyEncoding()
    train_freq = pd.Series(['A', 'A', 'B']) # A: 0.66, B: 0.33
    test_freq = pd.Series(['B'])
    
    res_train_f = func_freq(train_freq)
    b_freq = res_train_f.iloc[2]
    
    res_test_f = func_freq(test_freq)
    assert res_test_f.iloc[0] == approx(b_freq)

def test_transformer_storage_exposure():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'B': ['cat', 'dog'] * 10,
        'target': rng.rand(20)
    })
    
    # Force the use of cat_codes only to guarantee we see it
    synth = ft.GeneticFeatureSynthesis(
        num_features=1,
        population_size=10,
        max_generations=1,
        functions=['cat_codes'],
        n_jobs=1
    )
    synth.fit(df[['B']], df['target'])
    
    transformers = synth.get_feature_transformers()
    assert len(transformers) > 0
    # The first feature should have a list of states
    first_feature = list(transformers.keys())[0]
    states = transformers[first_feature]
    
    # Find the cat_codes state
    cat_codes_state = next((s for s in states if s['op'] == 'cat_codes'), None)
    assert cat_codes_state is not None
    assert isinstance(cat_codes_state['state'], dict)
    assert len(cat_codes_state['state']) >= 2
    # Verify we have some valid keys (could be 'cat'/'dog' or 0/1 if nested)
    keys = list(cat_codes_state['state'].keys())
    assert any(k in ['cat', 'dog'] or isinstance(k, (int, np.integer)) for k in keys)

def test_target_encoding():
    func = ft.synthesis.symbolic_functions.SymbolicTargetEncoding()
    X = pd.Series(['A', 'A', 'B', 'B', 'C'])
    y = pd.Series([10.0, 10.0, 20.0, 20.0, 30.0])
    
    # In fit mode (passing y)
    res_train = func(X, y=y)
    assert res_train.iloc[0] == approx(10.0, rel=0.2) # TargetEncoder smoothed mean
    assert res_train.iloc[2] == approx(20.0, rel=0.2)
    
    # In transform mode (y=None)
    test_X = pd.Series(['A', 'B'])
    res_test = func(test_X)
    assert res_test.iloc[0] == approx(res_train.iloc[0])
    assert res_test.iloc[1] == approx(res_train.iloc[2])

def test_symbolic_combine():
    func = ft.synthesis.symbolic_functions.SymbolicCombine()
    a = pd.Series(['A', 'B'])
    b = pd.Series(['1', '2'])
    expected = pd.Series(['A_1', 'B_2'])
    pd.testing.assert_series_equal(func(a, b), expected)

def test_symbolic_pair_freq():
    func = ft.synthesis.symbolic_functions.SymbolicPairFrequency()
    a = pd.Series(['A', 'A', 'B'])
    b = pd.Series(['1', '1', '2'])
    # Pairs: A_1, A_1, B_2. Freqs: A_1=2/3, B_2=1/3
    res = func(a, b)
    assert res.iloc[0] == approx(2/3)
    assert res.iloc[2] == approx(1/3)

def test_symbolic_pair_target_encoding():
    func = ft.synthesis.symbolic_functions.SymbolicPairTargetEncoding()
    a = pd.Series(['A', 'A', 'B', 'B', 'C'])
    b = pd.Series(['1', '1', '2', '2', '3'])
    y = pd.Series([10.0, 10.0, 20.0, 20.0, 30.0])
    
    res = func(a, b, y=y)
    assert res.iloc[0] == approx(10.0, rel=0.2)
    assert res.iloc[2] == approx(20.0, rel=0.2)

def test_symbolic_group_mean():
    func = ft.synthesis.symbolic_functions.SymbolicGroupMean()
    cat = pd.Series(['A', 'A', 'B'])
    num = pd.Series([10.0, 20.0, 100.0])
    # Group A mean = 15.0, Group B mean = 100.0
    res = func(cat, num)
    assert res.iloc[0] == 15.0
    assert res.iloc[2] == 100.0

def test_symbolic_group_diff():
    func = ft.synthesis.symbolic_functions.SymbolicGroupDifference()
    cat = pd.Series(['A', 'A', 'B'])
    num = pd.Series([10.0, 20.0, 100.0])
    # Mean A=15, B=100. Diffs: 10-15=-5, 20-15=5, 100-100=0
    res = func(cat, num)
    assert res.iloc[0] == -5.0
    assert res.iloc[1] == 5.0
    assert res.iloc[2] == 0.0

def test_symbolic_cat_indicator_mult():
    func = ft.synthesis.symbolic_functions.SymbolicCatIndicatorMultiply()
    cat = pd.Series(['A', 'B', 'A'])
    num = pd.Series([10.0, 20.0, 30.0])
    res = func(cat, num)
    # It picks one category randomly, say A. Then results: 10, 0, 30.
    # Or B. Then results: 0, 20, 0.
    assert (res != 0).any()
    assert (res.iloc[0] == 10.0 or res.iloc[1] == 20.0)

def test_symbolic_hashing_interaction():
    func = ft.synthesis.symbolic_functions.SymbolicHashingInteraction()
    a = pd.Series(['A', 'B'])
    b = pd.Series(['1', '2'])
    res = func(a, b)
    assert pd.api.types.is_numeric_dtype(res)
    assert len(res) == 2

def test_categorical_dtype_robustness():
    # Test that SymbolicCatCodes and SymbolicFrequencyEncoding
    # don't crash when passed a pandas.Categorical dtype (common in Kaggle)
    cat_data = pd.Series(pd.Categorical(['apple', 'banana', 'apple']))
    
    # Test CatCodes
    func_codes = ft.synthesis.symbolic_functions.SymbolicCatCodes()
    res_codes = func_codes(cat_data)
    assert len(res_codes) == 3
    assert not res_codes.isna().any()
    
    # Test Frequency
    func_freq = ft.synthesis.symbolic_functions.SymbolicFrequencyEncoding()
    res_freq = func_freq(cat_data)
    assert len(res_freq) == 3
    assert res_freq.iloc[0] == approx(2/3)

    # Test with new categories during transform phase
    new_cat_data = pd.Series(pd.Categorical(['cherry']))
    res_codes_new = func_codes(new_cat_data)
    # Should fill with -1 without crashing
    assert res_codes_new.iloc[0] == -1



