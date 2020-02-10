import numpy as np
import diffprivlib.models as dp
from tiresias.core.mechanisms import approximate_bounds

class LinearRegression(dp.LinearRegression):

    def fit(self, X, y, sample_weight=None):
        # TODO: concat X and y for norm, specify ranges
        if not self.data_norm:
            self.epsilon /= 2.0
            row_norms = np.linalg.norm(X, axis=1)
            _, max_norm = approximate_bounds(row_norms, self.epsilon)
            self.data_norm = max_norm
            for i in range(X.shape[0]):
                if np.linalg.norm(X[i]) > self.data_norm:
                    X[i] = X[i] * (self.data_norm - 1e-5) / np.linalg.norm(X[i])
        return super().fit(X, y, sample_weight=sample_weight)
