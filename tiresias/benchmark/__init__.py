"""
The `tiresias.benchmark` module provides an integrated framework for measuring
the performance of different differential privacy implementations on popular 
public-domain datasets.
"""
import os
import time
import numpy as np
import pandas as pd
import tiresias.core.mechanisms as mechanisms
import tiresias.core.machine_learning as ml

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype, load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def apply_ldp(X, y, epsilon):
    """
    Create a differentially private version of a dataset where X contains real 
    values and y contains discrete values.
    """
    X, y = X.copy(), y.copy()
    epsilon = epsilon / (X.shape[1] + 1)
    for dim in range(0, X.shape[1]):
        low, high = np.min(X[:,dim]), np.max(X[:,dim])
        X[:,dim] = mechanisms.bounded_continuous(X[:,dim], low=low, high=high, epsilon=epsilon)
    y = mechanisms.finite_categorical(y, set(y), epsilon=epsilon)
    return X, y

def evaluate(X_train, X_test, y_train, y_test, model_builder=None, N=20):
    accuracy = []
    start = time.time()
    for _ in range(N):
        model = model_builder()
        model.fit(X_train, y_train)
        accuracy.append(model.score(X_test, y_test))
    return np.mean(accuracy), (time.time() - start) / N

def benchmark_classification(data_loader=fetch_covtype):
    results = []
    X, y = data_loader(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Bounded Queries
    for epsilon in [10.0, 100.0]:
        X_dp, y_dp = apply_ldp(X_train, y_train, epsilon)
        for model in [SVC, LogisticRegression, RandomForestClassifier]:
            accuracy, running_time = evaluate(X_dp, X_test, y_dp, y_test, model_builder=model)
            results.append({
                "query": "bounded",
                "model": model.__name__,
                "epsilon": epsilon,
                "accuracy": accuracy,
                "running_time": running_time
            })
            print(results[-1])

    # Machine Learning Queries
    for epsilon in [10.0, 100.0]:
        for model in [ml.GaussianNB, ml.LogisticRegression]:
            accuracy, running_time = evaluate(X_dp, X_test, y_dp, y_test, model_builder=lambda: model(epsilon=epsilon))
            results.append({
                "query": "machine_learning",
                "model": model.__name__,
                "epsilon": epsilon,
                "accuracy": accuracy,
                "running_time": running_time
            })
            print(results[-1])

    # Federated Learning: Logistic Regression
    # ...

    # Federated Learning: Multilayer Perceptron
    # ...
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    __dir__ = os.path.dirname(__file__)

    df = benchmark_classification(load_breast_cancer)
    df.to_csv(os.path.join(__dir__, "breast_cancer.csv"), index=False)
    
    #df = benchmark_classification(fetch_covtype)
    #df.to_csv(os.path.join(__dir__, "covtype.csv"), index=False)
