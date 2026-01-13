
import featuristic as ft
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

def test_synthesizer_recovers_simple_formula():
    """
    Test that the synthesizer can recover a known simple relationship: y = A + B
    """
    np.random.seed(42)
    X = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.random.rand(100) # Noise column
    })
    # Target is exactly A + B
    y = X['A'] + X['B']
    
    # Restrict functions to make it easier/faster for the test to find the exact solution
    synth = ft.GeneticFeatureSynthesis(
        num_features=5,
        population_size=50,
        max_generations=10,
        functions=['add', 'subtract', 'mult'],
        n_jobs=1,
        pbar=False
    )
    
    synth.fit(X, y)
    features = synth.transform(X)
    
    # We expect one of the generated features to be perfectly correlated with y
    # or essentially identical to y.
    # The generated features start after the original ones (A, B, C)
    generated_data = features.iloc[:, 3:]
    
    # Check if any feature has > 0.99 correlation with y
    correlations = []
    for col in generated_data.columns:
        corr = np.corrcoef(generated_data[col], y)[0, 1]
        correlations.append(abs(corr))
        
    assert max(correlations) > 0.99, f"Failed to recover A+B relationship. Max correlation: {max(correlations)}"

def test_max_samples_subsampling():
    """
    Test that max_samples parameter works and actually runs on a subset.
    We can't easily peek inside to see if it subsampled without mocking, 
    but we can verify it runs successfully on data larger than max_samples.
    """
    np.random.seed(42)
    # Create 100 rows, set max_samples to 20
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.rand(100))
    
    synth = ft.GeneticFeatureSynthesis(
        num_features=2,
        population_size=10,
        max_generations=2,
        max_samples=20, # Should trigger subsampling logic
        n_jobs=1,
        pbar=False
    )
    
    synth.fit(X, y)
    res = synth.transform(X)
    assert res.shape[0] == 100 # Output should still be full size
    assert res.shape[1] > 5    # Should have generated features

def test_parallel_execution_consistency():
    """
    Test that running in parallel produces valid results (structurally).
    Exact reproducibility between serial/parallel depends on seeding and joblib,
    here we just ensure it completes and produces features.
    """
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(50, 4), columns=['a','b','c','d'])
    y = X['a'] * X['b'] # Simple target
    
    synth = ft.GeneticFeatureSynthesis(
        num_features=3,
        population_size=20,
        max_generations=5,
        n_jobs=-1, # Parallel
        pbar=False
    )
    
    synth.fit(X, y)
    res = synth.transform(X)
    assert res.shape[1] > 4
    
def test_custom_functions_restriction():
    """
    Test that providing a custom list of functions works and doesn't crash.
    """
    X = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    y = pd.Series([10,20,30])
    
    # Only allow 'sin'
    synth = ft.GeneticFeatureSynthesis(
        num_features=2,
        population_size=10,
        max_generations=2,
        functions=['sin'],
        n_jobs=1,
        pbar=False
    )
    
    synth.fit(X, y)
    # If it works, it passes. If it tries to use 'add' (default), it strictly shouldn't fail 
    # unless 'add' is somehow required, but the synthesizer should respect the list.
    # To verifying it ONLY used sin would require parsing the formulas, 
    # but checking it fits is a good start.
    assert synth.fit_called

def test_get_feature_info_structure():
    """
    Test that get_feature_info returns the expected dataframe structure.
    """
    # Increase size/variance so it finds something
    X = pd.DataFrame({'a': np.random.rand(20), 'b': np.random.rand(20)})
    y = X['a'] * 2 # strong signal
    
    synth = ft.GeneticFeatureSynthesis(
        num_features=1,
        population_size=10,
        max_generations=2,
        n_jobs=1,
        pbar=False
    )
    synth.fit(X, y)
    
    info = synth.get_feature_info()
    assert isinstance(info, pd.DataFrame)
    assert "name" in info.columns
    assert "formula" in info.columns
    assert "fitness" in info.columns
    assert len(info) > 0
