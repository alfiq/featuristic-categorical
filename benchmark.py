
import time
import pandas as pd
import numpy as np
import featuristic as ft
from sklearn.datasets import make_regression

def benchmark():
    # Generate massive dataset: 1 million rows
    print("Generating 1,000,000 rows of data...")
    X_np, y_np = make_regression(n_samples=1000000, n_features=20, random_state=42)
    X = pd.DataFrame(X_np, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y_np)

    print("\nStarting Genetic Synthesis on 1M rows...")
    print("(Using max_samples=20000 for evolution to demonstrate smart scaling)")
    
    synth = ft.GeneticFeatureSynthesis(
        num_features=5,
        population_size=100,
        max_generations=5,
        n_jobs=-1,            # Use all cores
        max_samples=20000,    # Smart scaling
        parsimony_coefficient=0.01,
        verbose=True
    )

    start_time = time.time()
    synth.fit(X, y)
    end_time = time.time()
    
    print(f"\nTraining completed in: {end_time - start_time:.4f} seconds")
    
    print("\nTransforming full 1M row dataset...")
    t_start = time.time()
    res = synth.transform(X)
    t_end = time.time()
    print(f"Transform completed in: {t_end - t_start:.4f} seconds")
    print(f"Final output shape: {res.shape}")

if __name__ == "__main__":
    benchmark()
