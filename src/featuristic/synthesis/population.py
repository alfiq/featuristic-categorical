""" The population module contains the classes for the population of programs in the
genetic programming algorithm. """

from copy import deepcopy
from typing import Callable, List, Self

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from .program import random_prog, select_random_node, node_count, render_prog


class BasePopulation:
    """
    A class to represent the population of symbolic programs in the
    genetic programming algorithm.
    """

    def __init__(
        self,
        population_size: int,
        operations: List,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
    ):
        """
        Initialize the population.

        Args
        ----
        population_size : int
            The size of the population. The larger the population, the more
            likely the algorithm will find a good solution, but the longer it
            will take to run.

        operations : list
            The list of functions to use in the programs. These are the
            functions that the algorithm can use to create the programs with.
        """
        self.population_size = population_size
        self.operations = operations
        self.population = None
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self._cache = {}

    def initialize(self, X: pd.DataFrame) -> Self:
        """
        Setup the initial population with simple, random programs.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.
        """
        self.population = [
            random_prog(0, X, self.operations) for _ in range(self.population_size)
        ]
        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series = None) -> List[pd.Series]:
        """
        Evaluate the population against the dataframe of features.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.
        y : pd.Series, optional
            The target variable. Used for stateful transformations.

        return
        ------
        list
            The predicted values.
        """
        raise NotImplementedError

    def compute_fitness(
        self,
        fitness_func: Callable,
        parsimony_coefficient: float,
        prediction,
        y: pd.Series,
    ) -> List[float]:
        """
        Compute the fitness of the population.

        Args
        ----

        fitness_func : callable
            The fitness function to use.

        parsimony_coefficient : float
            The parsimony coefficient.

        prediction : list
            The predicted values.

        y : pd.Series
            The true values.

        return
        ------
        list
            The fitness of the population.
        """
        raise NotImplementedError

    def _get_prog_signature(self, node: dict):
        """
        Get a hashable signature for the program node.
        """
        if "children" not in node:
            return node["feature_name"]
        
        # Use function name or the function object hash if name distinctness is insufficient
        # func.name strings are better for readability and usually sufficient
        fname = node["func"].name
        child_sigs = tuple(self._get_prog_signature(c) for c in node["children"])
        return (fname, child_sigs)

    def _evaluate_program(self, node: dict, X_data: dict, y=None) -> np.ndarray:
        """
        Evaluate the program against the dataframe of features.

        Args
        ----
        node : dict
            The program to evaluate.

        X_data : dict
            Dictionary of numpy arrays for features.

        y : pd.Series, optional
            The target variable.

        return
        ------
        np.ndarray
            The predicted values.
        """
        # Memoization check
        # Use fast tuple signature
        prog_sig = self._get_prog_signature(node)
        
        if prog_sig in self._cache:
            return self._cache[prog_sig]

        if "children" not in node:
            # Leaf node (feature)
            res = X_data[node["feature_name"]]
        else:
            # Recursive evaluation
            args = [self._evaluate_program(c, X_data, y) for c in node["children"]]
            func_obj = node["func"]

            # If the function is stateful and accepts a target, pass it
            if hasattr(func_obj, "requires_target") and func_obj.requires_target:
                res = func_obj(*args, y=y)
            else:
                res = func_obj(*args)

        # Cache the result
        self._cache[prog_sig] = res
        return res

    def _evaluate_df(self, node: dict, X: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        """
        Legacy wrapper for compatibility or single-program checks.
        Uses the optimized implementation.
        """
        # Convert X to dict of numpy arrays for speed
        X_data = {col: X[col].values for col in X.columns}
        res = self._evaluate_program(node, X_data, y)
        if isinstance(res, pd.Series):
            return res
        return pd.Series(res)

    def _get_random_parent(self, fitness: List[float]) -> dict:
        """
        Select a random parent from the population using tournament selection.

        Args
        ----
        fitness : list
            The fitness values of the population.

        return
        ------
        dict
            The selected parent program.
        """
        tournament_members = [
            np.random.randint(0, self.population_size)
            for _ in range(self.tournament_size)
        ]
        member_fitness = [(fitness[i], self.population[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def _crossover(self, selected1: dict, selected2: dict) -> dict:
        """
        Perform crossover mutation between two selected programs.

        Args
        ----
        selected1 : dict
            The first selected program.

        selected2 : dict
            The second selected program.

        return
        ------
        dict
            The offspring program.
        """
        offspring = deepcopy(selected1)
        xover_point1 = select_random_node(offspring, None, 0)
        xover_point2 = select_random_node(selected2, None, 0)
        child_count = len(xover_point1["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count)
        xover_point1["children"][child_idx] = xover_point2
        return offspring

    def _mutate(self, selected: dict, X: pd.DataFrame) -> dict:
        """
        Mutate the selected program by replacing a random node with a new random program.

        Args
        ----
        selected : dict
            The selected program to mutate.

        X : pd.DataFrame
            The dataframe with the features.
        """
        offspring = deepcopy(selected)
        mutate_point = select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count)
        mutate_point["children"][child_idx] = random_prog(0, X, self.operations)
        return offspring

    def _get_offspring(self, fitness: List[float], X: pd.DataFrame) -> dict:
        """
        Get the offspring of two parents using crossover mutation.

        Args
        ----
        fitness : list
            The fitness values of the population.

        X : pd.DataFrame
            The dataframe with the features.

        return
        ------
        dict
            The offspring program.
        """
        parent1 = self._get_random_parent(fitness)
        if np.random.uniform() < self.crossover_prob:
            parent2 = self._get_random_parent(fitness)
            return self._crossover(parent1, parent2)

        return self._mutate(parent1, X)

    def evolve(self, fitness: List[float], X: pd.DataFrame) -> Self:
        """
        Evolve the population by creating a new generation of programs.

        Args
        ----
        fitness : list
            The fitness values of the population.

        X : pd.DataFrame
        """
        self.population = [
            self._get_offspring(fitness, X) for _ in range(self.population_size)
        ]
        return self


class SerialPopulation(BasePopulation):
    """
    A class to represent the population of programs in the genetic programming algorithm where
    the programs are evaluated serially.
    """

    def __init__(
        self,
        population_size: int,
        operations: List,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
    ):
        """
        Initialize the population class.

        Args
        ----

        population_size : int
            The size of the population.

        operations : list
            The list of functions to use in the programs.
        """
        super().__init__(population_size, operations, tournament_size, crossover_prob)

    def evaluate(self, X: pd.DataFrame, y: pd.Series = None) -> List[np.ndarray]:
        """
        Evaluate the population against the current program.
        """
        # Clear cache at start of evaluation if the data X is different?
        # Actually, if we fit, X is same. If we transform new data, X is different.
        # We should clear cache to be safe, or manage it smarter. 
        # For simplicity and safety across calls with different data: clear it.
        self._cache = {}
        
        # Pre-convert data to numpy for speed
        X_data = {col: X[col].values for col in X.columns}
        
        results = []
        for prog in self.population:
            res = self._evaluate_program(prog, X_data, y)
            results.append(res)
            
        return results

    def compute_fitness(
        self,
        fitness_func: Callable,
        parsimony_coefficient: float,
        prediction,
        y: pd.Series,
    ) -> List[float]:
        """
        Compute the fitness of the population.

        Args
        ----

        fitness_func : callable
            The fitness function to use.

        parsimony_coefficient : float
            The parsimony coefficient.

        prediction : list
            The predicted values.

        y : pd.Series
            The true values.

        return
        ------
        list
            The fitness of the population.
        """
        score = [
            fitness_func(prog, parsimony_coefficient, y, pred)
            for prog, pred in zip(self.population, prediction)
        ]
        return score

    def apply_parsimony(
        self, scores: List[float], parsimony_coefficient: float
    ) -> List[float]:
        """
        Apply the parsimony coefficient to the fitness scores.

        Args
        ----
        scores : list
            The fitness scores.

        parsimony_coefficient : float
            The parsimony coefficient.

        return
        ------
        list
            The updated fitness scores.
        """

        def _parsimony_coefficient(loss, prog):
            penalty = node_count(prog) ** parsimony_coefficient
            loss = loss / penalty
            return -loss

        return [
            _parsimony_coefficient(loss, prog)
            for loss, prog in zip(scores, self.population)
        ]


class ParallelPopulation(BasePopulation):
    """
    A class to represent the population of programs in the genetic programming algorithm where
    the programs are evaluated in parallel via joblib.
    """

    def __init__(
        self,
        population_size: int,
        operations: List,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
        n_jobs: int = -1,
    ):
        """
        Initialize the population class.

        Args
        ----

        population_size : int
            The size of the population.

        operations : list
            The list of functions to use in the programs.

        n_jobs : int
            The number of jobs to use in the parallel evaluation.
        """
        super().__init__(population_size, operations, tournament_size, crossover_prob)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def evaluate(self, X: pd.DataFrame, y: pd.Series = None) -> List[pd.Series]:
        """
        Evaluate the population against the current program. This is done in parallel.
        """
        # Parallel evaluation is tricky with caching.
        # We'll use the optimized _evaluate_program but without shared caching across processes.
        # However, we still benefit from the X_data conversion/numpy speedup within each task.
        
        # Note: Sending X_data (dict of arrays) is likely faster/same as DataFrame.
        X_data = {col: X[col].values for col in X.columns}
        
        # We can't easily populate self._cache here from workers.
        # But we can pass an empty local cache to _evaluate_program if we modified it to accept one or use instance.
        # Since _evaluate_program uses self._cache, and 'self' is pickled, 
        # each worker gets a copy of self. So they have their own local cache for the duration of that individual's eval.
        # This helps if an individual has repeated subtrees (common).
        
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_wrapper)(prog, X_data, y) for prog in self.population
        )

    def _evaluate_wrapper(self, prog, X_data, y):
        # Helper to ensure we wrap back to Series for consistency with rest of pipeline expectations
        # and manage local cache state for the worker
        self._cache = {} # Reset local cache for this worker/task
        res = self._evaluate_program(prog, X_data, y)
        return res

    def compute_fitness(
        self,
        fitness_func: Callable,
        parsimony_coefficient: float,
        prediction,
        y: pd.Series,
    ) -> List[float]:
        """
        Compute the fitness of the population.

        Args
        ----

        fitness_func : callable
            The fitness function to use.

        parsimony_coefficient : float
            The parsimony coefficient.

        prediction : list
            The predicted values.

        y : pd.Series
            The true values.

        return
        ------
        list
            The fitness of the population.
        """
        return Parallel(n_jobs=cpu_count())(
            delayed(fitness_func)(prog, parsimony_coefficient, y, pred)
            for prog, pred in zip(self.population, prediction)
        )

    def apply_parsimony(
        self, scores: List[float], parsimony_coefficient: float
    ) -> List[float]:
        """
        Apply the parsimony coefficient to the fitness scores.

        Args
        ----
        scores : list
            The fitness scores.

        parsimony_coefficient : float
            The parsimony coefficient.

        return
        ------
        list
            The updated fitness scores.
        """

        def _parsimony_coefficient(loss, prog):
            penalty = node_count(prog) ** parsimony_coefficient
            loss = loss / penalty
            return -loss

        return Parallel(n_jobs=cpu_count())(
            delayed(_parsimony_coefficient)(score, prog)
            for score, prog in zip(scores, self.population)
        )
