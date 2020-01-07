import argparse
import time
import warnings
import logging
import numpy as np
import pandas as pd

from tiresias.core import machine_learning as ml
from tiresias.benchmark.utils import make_ldp
from tiresias.benchmark.utils import FederatedLearningClassifier

from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.simplefilter(action='ignore')
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def run(X, y, model, epsilon, delta, use_ldp):
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    if use_ldp:
        X_train, y_train = make_ldp(X_train, y_train, epsilon, delta)
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
    X = RobustScaler().fit_transform(X)

    results = []
    for epsilon in [10.0, 100.0, 1000.0]:
        # Bounded Queries
        for model in [LogisticRegression(), RandomForestClassifier(), LinearSVC()]:
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
        for model in [ml.LogisticRegression(epsilon=epsilon), ml.GaussianNB(epsilon=epsilon)]:
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
        for model in [FederatedLearningClassifier(epsilon=epsilon, delta=1.0 / len(X), epochs=32, lr=0.01)]:
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

def report(csv):
    datasets = [
        ("Wine", load_wine(return_X_y=True)),
        ("Breast Cancer", load_breast_cancer(return_X_y=True)),
        ("Forest Cover", fetch_covtype(return_X_y=True)),
        ("KDDCup99", fetch_kddcup99(return_X_y=True))
    ]

    dfs = []
    for dataset, (X, y) in datasets:
        log.info("%s: X = %s, Y = %s" % (dataset, X.shape, y.shape))
        df = benchmark(X, y)
        df["dataset"] = dataset
        dfs.append(df)
        pd.concat(dfs).to_csv(csv)
    df = pd.concat(dfs)

    df = df.set_index(["dataset", "epsilon", "type", "model"])
    df.to_csv(args.csv)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="classification.csv")
    args = parser.parse_args()
    df = report(args.csv)
    print(df)
