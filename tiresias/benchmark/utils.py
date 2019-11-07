"""
The `tiresias.benchmark.utils` module provides utility functions to help train 
and evaluate differentially private models.
"""
import numpy as np
import tiresias.core.mechanisms as mechanisms

def apply_ldp(X, y, epsilon, is_discrete=False):
    """
    This function produces an epsilon-differentially private version of the 
    given data where X consists of real values and y contains either real 
    or discrete values.
    """
    X, y = X.copy(), y.copy()
    epsilon = epsilon / (X.shape[1] + 1)
    for dim in range(0, X.shape[1]):
        low, high = np.min(X[:,dim]), np.max(X[:,dim])
        X[:,dim] = mechanisms.bounded_continuous(X[:,dim], low=low, high=high, epsilon=epsilon)
    if is_discrete:
        y = mechanisms.finite_categorical(y, set(y), epsilon=epsilon)
    else:
        low, high = np.min(y), np.max(y)
        y = mechanisms.bounded_continuous(y, low=low, high=high, epsilon=epsilon)
    return X, y
