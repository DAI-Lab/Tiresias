import numpy as np
import tiresias.core.mechanisms as mechanisms

def _ldp(x, epsilon, delta, continuous=True):
    if continuous:
        low, high = np.min(x), np.max(x)
        return mechanisms.bounded_continuous(x, low=low, high=high, epsilon=epsilon, delta=delta)
    else:
        return mechanisms.finite_categorical(x, set(x), epsilon=epsilon, delta=delta)

def make_ldp(X, y, epsilon, delta):
    num_rows, num_cols = X.shape
    assert X.shape[0] == y.shape[0]

    X, y = X.copy(), y.copy()
    for col_idx in range(0, num_cols):
        X[:,col_idx] = _ldp(X[:,col_idx], epsilon / (2 * X.shape[1]), delta / (2 * X.shape[1]))
    y = _ldp(y, epsilon / 2.0, delta / 2.0, continuous=False)
    return X, y
