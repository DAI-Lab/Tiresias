import numpy as np
import diffprivlib.models as dp
from tiresias.core.mechanisms import approximate_bounds

class GaussianNB(dp.GaussianNB):

    def __init__(self, epsilon=1, bounds=None, priors=None, var_smoothing=1e-9):
        super().__init__(epsilon, bounds, priors, var_smoothing)

    def fit(self, X, y, sample_weight=None):
        if not self.bounds:
            self.bounds = []
            self.epsilon /= 2.0
            for column in range(X.shape[1]):
                bounds = approximate_bounds(X[:,column], self.epsilon / X.shape[1])
                self.bounds.append(bounds)
                X[:,column] = np.minimum(np.maximum(X[:,column], bounds[0]), bounds[1])
        return super().fit(X, y, sample_weight=sample_weight)

class LogisticRegression(dp.LogisticRegression):

    def fit(self, X, y, sample_weight=None):
        if not self.data_norm:
            assert self.epsilon > 1.0
            self.epsilon -= 1.0
            row_norms = np.linalg.norm(X, axis=1)
            _, max_norm = approximate_bounds(row_norms, 1.0)
            self.data_norm = max_norm
            for i in range(X.shape[0]):
                if np.linalg.norm(X[i]) > self.data_norm:
                    X[i] = X[i] * (self.data_norm - 1e-5) / np.linalg.norm(X[i])
        return super().fit(X, y, sample_weight=sample_weight)

class TiresiasClassifier(dp.LogisticRegression):

    def __init__(self, epsilon):
        self.epsilon_model = epsilon * 0.5
        self.epsilon_selection = epsilon * 0.5

    def fit(self, X, y):
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        models, scores = [], []
        model = GaussianNB(epsilon=self.epsilon_model)
        models.append(model.fit(X_train, y_train))
        scores.append(f1_score(y_test, model.predict(X_test)))
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = LogisticRegression(epsilon=self.epsilon_model, C=C)
            models.append(model.fit(X_train, y_train))
            scores.append(f1_score(y_test, model.predict(X_test)))
        probabilities = np.exp(self.epsilon_selection * np.array(scores) / 2)
        probabilities = probabilities / np.sum(probabilities)

        self.model = np.random.choice(models, p=probabilities, size=1)[0]
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
