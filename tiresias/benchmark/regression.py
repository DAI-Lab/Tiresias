import argparse
import time
import warnings
import logging
import numpy as np
import pandas as pd

from tiresias.core import machine_learning as ml
from tiresias.benchmark.utils import make_ldp
from tiresias.benchmark.utils import FederatedLearningRegressor

from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

warnings.simplefilter(action='ignore')
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def run(X, y, model, epsilon, delta, use_ldp):
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    if use_ldp:
        X_train, y_train = make_ldp(X_train, y_train, epsilon, delta, classification=False)
    model.fit(X_train, y_train)
    
    running_time = time.time() - start
    return model.score(X_test, y_test), running_time

def run_N(X, y, model, epsilon, delta, N=20, use_ldp=True):
    accuracies, running_times = [], []
    for _ in range(N):
        accuracy, running_time = run(X, y, model, epsilon, delta, use_ldp)
        accuracies.append(accuracy)
        running_times.append(running_time)
    return np.mean(accuracies), np.mean(running_times)

def benchmark(X, y):
    results = []
    for epsilon in [10.0, 100.0, 1000.0]:
        # Bounded Queries
        for model in [LinearRegression(), RandomForestRegressor(), SVR()]:
            accuracy, running_time = run_N(X, y, model, epsilon=epsilon, delta=False, use_ldp=True)
            results.append({
                "type": "bounded",
                "model": type(model).__name__,
                "epsilon": epsilon,
                "accuracy": accuracy,
                "running_time": running_time
            })
            log.info("%s" % results[-1])

        # Machine Learning Queries
        for model in [ml.LinearRegression(epsilon=epsilon)]:
            accuracy, running_time = run_N(X, y, model, epsilon=epsilon, delta=False, use_ldp=False)
            results.append({
                "type": "machine_learning",
                "model": type(model).__name__,
                "epsilon": epsilon,
                "accuracy": accuracy,
                "running_time": running_time
            })
            log.info("%s" % results[-1])

        # Federated Learning Queries
        for model in [FederatedLearningRegressor(epsilon=epsilon, delta=1.0 / len(X), epochs=32, lr=0.01)]:
            accuracy, running_time = run_N(X, y, model, epsilon=epsilon, delta=False, use_ldp=False)
            results.append({
                "type": "federated_learning",
                "model": type(model).__name__,
                "epsilon": epsilon,
                "accuracy": accuracy,
                "running_time": running_time
            })
            log.info("%s" % results[-1])

    return pd.DataFrame(results)

def report():
    datasets = [
        ("Boston Housing", load_boston(return_X_y=True)),
        ("Diabetes Progression", load_diabetes(return_X_y=True)),
        ("California Housing", fetch_california_housing(return_X_y=True)),
    ]

    dfs = []
    for dataset, (X, y) in datasets:
        log.info("%s: X = %s, Y = %s" % (dataset, X.shape, y.shape))
        df = benchmark(X, y)
        df["dataset"] = dataset
        dfs.append(df)
    df = pd.concat(dfs)

    return df.set_index(["dataset", "epsilon", "type", "model",])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="classification.csv")
    args = parser.parse_args()

    df = report()
    df.to_csv(args.csv)
    print(df)
