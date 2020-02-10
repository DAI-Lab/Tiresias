import torch
import numpy as np
from tqdm import tqdm
from sklearn.base import ClassifierMixin, RegressorMixin

import tiresias.core.mechanisms as mechanisms
from tiresias.core.gradients import get_gradients, put_gradients, merge_gradients

def _ldp(x, epsilon, delta, continuous=True):
    if continuous:
        low, high = np.min(x), np.max(x)
        return mechanisms.bounded_continuous(x, low=low, high=high, epsilon=epsilon)
    else:
        return mechanisms.finite_categorical(x, set(x), epsilon=epsilon)

def make_ldp(X, y, epsilon, delta, classification=True):
    num_rows, num_cols = X.shape
    assert X.shape[0] == y.shape[0]

    p = 0.9 # use 90% of budget for X, 10% for Y
    X, y = X.copy(), y.copy()
    for col_idx in range(0, num_cols):
        X[:,col_idx] = _ldp(X[:,col_idx], p * epsilon / X.shape[1], p * delta / X.shape[1])
    y = _ldp(y, (1.0 - p) * epsilon, (1.0 - p) * delta, continuous=not classification)
    return X, y

class FederatedLearningWrapper(object):

    def __init__(self, model, loss, epsilon, delta, epochs, lr, batch_size):
        self.model = model
        self.epsilon = epsilon
        self.delta = delta
        self.epochs = epochs
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X, Y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epsilon = self.epsilon / self.epochs
        delta = self.delta / self.epochs
        for epoch in tqdm(range(self.epochs)):
            gradients = []
            for i in range(len(X)):
                self.model.zero_grad()
                x = torch.FloatTensor(X[i]).unsqueeze(0)
                y = torch.tensor(Y[i]).unsqueeze(0)
                loss = self.loss(self.model(x), y)
                loss.backward()
                gradients.append(get_gradients(self.model, epsilon, delta))
                if len(gradients) > self.batch_size:
                    put_gradients(self.model, merge_gradients(gradients))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    gradients = []

    def predict(self, X):
        Y = []
        for i in range(len(X)):
            x = torch.FloatTensor(X[i]).unsqueeze(0)
            y = self.model(x)[0]
            Y.append(y.detach().numpy())
        return np.stack(Y)

class FederatedLearningClassifier(ClassifierMixin):

    def __init__(self, epsilon, delta, epochs, lr):
        self.epsilon = epsilon
        self.delta = delta
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self._class_to_i = {k: i for i, k in enumerate(set(y))}
        self._i_to_class = {v: k for k, v in self._class_to_i.items()}

        self._model = FederatedLearningWrapper(
            model=torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], 16),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(16, len(self._class_to_i)),
            ),
            loss=torch.nn.functional.cross_entropy, 
            epsilon=self.epsilon, 
            delta=self.delta,
            epochs=self.epochs, 
            lr=self.lr,
            batch_size=128,
        )
        self._model.fit(X, np.array([self._class_to_i[k] for k in y]))

    def predict(self, X):
        y_pred = self._model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        return [self._i_to_class[i] for i in y_pred]

class FederatedLearningRegressor(RegressorMixin):

    def __init__(self, epsilon, delta, epochs, lr):
        self.epsilon = epsilon
        self.delta = delta
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self._model = FederatedLearningWrapper(
            model=torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], 16),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(16, 1),
            ),
            loss=torch.nn.functional.mse_loss, 
            epsilon=self.epsilon, 
            delta=self.delta,
            epochs=self.epochs, 
            lr=self.lr,
            batch_size=128,
        )
        self._model.fit(X, y)

    def predict(self, X):
        y_pred = self._model.predict(X)
        return y_pred

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from tiresias.core import machine_learning as ml

    X, y = load_boston(return_X_y=True)

    for epsilon in [100.0]:

        clf = ml.LinearRegression(epsilon=epsilon)
        clf.fit(X, y)
        print("ML (%s): %s" % (epsilon, clf.score(X, y)))

        clf = FederatedLearningRegressor(
            epsilon=epsilon,
            delta=1.0 / len(X),
            epochs=32,
            lr=0.01,
        )
        clf.fit(X, y)
        print("FL (%s): %s" % (epsilon, clf.score(X, y)))
