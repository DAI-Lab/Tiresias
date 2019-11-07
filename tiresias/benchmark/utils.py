"""
The `tiresias.benchmark.utils` module provides utility functions to help train 
and evaluate differentially private models.
"""
import torch
import numpy as np
import tiresias.core.mechanisms as mechanisms
from sklearn.metrics import r2_score
from tiresias.core.federated_learning import get_gradients, put_gradients, merge_gradients

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

class DreadNought(object):
    """
    The `DreadNought` class provides a sklearn-like wrapper around PyTorch 
    models which simulates federated learning with local differential privacy.

    ```
    model = DreadNought(
        model=torch.nn.Sequential(
            torch.nn.Linear(3, 1),
        ), 
        loss=torch.nn.functional.mse_loss, 
        epsilon=1000.0, 
        epochs=100, 
        lr=0.1
    )

    X = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    Y = np.array([3.0, 2.0, 2.0, 2.0, 0.0])
    model.fit(X, Y)
    print(model.score(X, Y))
    ```
    """

    def __init__(self, model, loss, epsilon, epochs, lr):
        self.model = model
        self.epsilon = epsilon
        self.epochs = epochs
        self.loss = loss
        self.lr = lr

    def fit(self, X, Y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epsilon = self.epsilon / self.epochs
        for epoch in range(self.epochs):
            gradients = []
            for i in range(X.shape[0]):
                self.model.zero_grad()
                x = torch.FloatTensor([X[i]])
                y = torch.FloatTensor([[Y[i]]])
                loss = self.loss(self.model(x), y)
                loss.backward()
                gradients.append(get_gradients(self.model, epsilon))
            put_gradients(self.model, merge_gradients(gradients))
            optimizer.step()

    def predict(self, X):
        Y = []
        for i in range(X.shape[0]):
            x = torch.FloatTensor([X[i]])
            y = self.model(x)
            Y.append(y.item())
        return np.array(Y)

    def score(self, X, Y):
        return r2_score(Y, self.predict(X))
