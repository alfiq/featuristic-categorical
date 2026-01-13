"""Fitness functions for measuring how well the features are performing"""

import warnings

import numpy as np
import pandas as pd
import scipy
import sys
from scipy.stats import pearsonr

from .program import node_count


def fitness_pearson(
    program: dict, parsimony: float, y_true: pd.Series, y_pred: pd.Series
):
    """
    Compute the fitness of a program based on the pearson correlation and the parsimony coefficient

    Args:

    program: dict
        The program to evaluate

    parsimony: float
        The parsimony coefficient

    y_true: pd.Series
        The true values

    y_pred: pd.Series
        The predicted values
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.stats.NearConstantInputWarning)
        
        # y_pred might be a numpy array now. Ensure efficient checks.
        if isinstance(y_pred, pd.Series):
             y_pred = y_pred.values
        if isinstance(y_true, pd.Series):
             y_true = y_true.values

        if not np.issubdtype(y_pred.dtype, np.number):
            # Try to convert to float (handle object arrays of numbers)
            try:
                y_pred = y_pred.astype(float)
            except (ValueError, TypeError):
                # If conversion fails (e.g. strings), it's invalid
                return sys.maxsize

        if not np.isfinite(y_pred).all():
             return sys.maxsize

        if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
            return sys.maxsize

        loss = abs(pearsonr(y_true, y_pred).statistic)
        
        penalty = node_count(program) ** parsimony
        loss /= penalty
        return -loss
