"""
The `tiresias.benchmark.regression` module evaluates a variety of different
regression models and datasets for different epsilon values.
"""
import time
import torch
import logging
import warnings
import numpy as np
import pandas as pd
from tiresias.core import machine_learning as ml
from tiresias.benchmark.utils import apply_ldp, DreadNought

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

warnings.simplefilter(action='ignore')
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

DATA_LOADERS = {
    "Diabetes Progression": lambda: load_diabetes(return_X_y=True),
    "California Housing Prices": lambda: fetch_california_housing(return_X_y=True),
}

def _run_sklearn(X_train, X_test, y_train, y_test, model_builder=None, N=10):
    accuracy, running_time = [], []
    for _ in range(N):
        start = time.time()
        model = model_builder()
        model.fit(X_train, y_train)
        accuracy.append(model.score(X_test, y_test))
        running_time.append(time.time() - start)
    return np.mean(accuracy), np.mean(running_time)

def dreadnought_test(input_dims, epsilon):
    def _builder():
        model = DreadNought(
            model=torch.nn.Sequential(
                torch.nn.Linear(input_dims, 1),
            ), 
            loss=torch.nn.functional.mse_loss, 
            epsilon=epsilon, 
            epochs=16, 
            lr=0.01
        )
        return model
    _builder.__name__ = "DreadNought.LinearRegression"
    return _builder

def run_benchmark(epsilons=[10.0, 100.0], N=10):
    """
    This function evaluates the pre-specified models and datasets with the 
    given epsilon values. Each model-dataset pair is evaluated N times for
    each epsilon value and the average accuracy and running time is returned.
    """
    results = []
    for dataset_name, data_loader in DATA_LOADERS.items():
        X, y = data_loader()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        log.info("Loaded the %s dataset..." % dataset_name)
        log.info("Train: X = %s, Y = %s" % (X_train.shape, y_train.shape))
        log.info("Test : X = %s, Y = %s" % (X_test.shape, y_test.shape))

        for epsilon in epsilons:
            # LDP Queries
            X_dp, y_dp = apply_ldp(X_train, y_train, epsilon, is_discrete=True)
            for model in [RandomForestRegressor, SVR, LinearRegression, dreadnought_test(input_dims=X_train.shape[1])]:
                log.info("Running %s (ε = %s)" % (model.__name__, epsilon))
                accuracy, running_time = _run_sklearn(X_dp, X_test, y_dp, y_test, model_builder=model)
                log.info("Done. Acc = %.02f, Speed = %.02f" % (accuracy, running_time))

                results.append({
                    "dataset": dataset_name,
                    "query": "bounded",
                    "model": model.__name__,
                    "epsilon": epsilon,
                    "accuracy": accuracy,
                    "running_time": running_time
                })

            # Federated Learning
            for model in [dreadnought_test(input_dims=X_train.shape[1], epsilon=epsilon)]:
                log.info("Running %s (ε = %s)" % (model.__name__, epsilon))
                accuracy, running_time = _run_sklearn(X_train, X_test, y_train, y_test, model_builder=model)
                log.info("Done. Acc = %.02f, Speed = %.02f" % (accuracy, running_time))

                results.append({
                    "dataset": dataset_name,
                    "query": "federated_learning",
                    "model": model.__name__,
                    "epsilon": epsilon,
                    "accuracy": accuracy,
                    "running_time": running_time
                })

    return pd.DataFrame(results)
