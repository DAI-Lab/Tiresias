import numpy as np
import tiresias.core.mechanisms as mechanisms

def _ldp(x, epsilon, delta, continuous=True):
    if continuous:
        low, high = np.min(x), np.max(x)
        return mechanisms.bounded_continuous(x, low=low, high=high, epsilon=epsilon, delta=delta)
    else:
        return mechanisms.finite_categorical(x, set(x), epsilon=epsilon, delta=delta)

def make_ldp(X, y, epsilon, delta, classification=True):
    num_rows, num_cols = X.shape
    assert X.shape[0] == y.shape[0]

    p = 0.5 # use 70% of budget for X, 30% for Y
    X, y = X.copy(), y.copy()
    for col_idx in range(0, num_cols):
        X[:,col_idx] = _ldp(X[:,col_idx], p * epsilon / X.shape[1], p * delta / X.shape[1])
    y = _ldp(y, (1.0 - p) * epsilon, (1.0 - p) * delta, continuous=not classification)
    return X, y
