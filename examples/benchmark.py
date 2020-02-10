"""
This script uses the benchmark module to evaluate the system on a variety of 
datasets with different privacy levels.
"""
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import *
from tiresias.benchmark import benchmark

np.random.seed(42)
torch.random.manual_seed(42)

if __name__ == "__main__":
    reports = []
    num_trials = 10
    detailed = False

    for _ in range(num_trials):

        for epsilon in [16.0, 32.0, 64.0]:
            
            X, y = load_wine(return_X_y=True)
            report = benchmark(X, y, epsilon=epsilon, delta=1e-5, problem_type="classification")
            if not detailed:
                report = report.groupby(["model", "type", "epsilon"]).agg({
                    "accuracy": lambda x: x.max(),
                    "time": lambda x: x.mean(),
                })
                report = report.reset_index()
            report["dataset"] = "Wine"
            report["problem_type"] = "classification"
            reports.append(report)

            X, y = load_breast_cancer(return_X_y=True)
            report = benchmark(X, y, epsilon=epsilon, delta=1e-5, problem_type="classification")
            if not detailed:
                report = report.groupby(["model", "type", "epsilon"]).agg({
                    "accuracy": lambda x: x.max(),
                    "time": lambda x: x.mean(),
                })
                report = report.reset_index()
            report["dataset"] = "Cancer"
            report["problem_type"] = "classification"
            reports.append(report)

            X, y = load_boston(return_X_y=True)
            report = benchmark(X, y, epsilon=epsilon, delta=1e-5, problem_type="regression")
            if not detailed:
                report = report.groupby(["model", "type", "epsilon"]).agg({
                    "accuracy": lambda x: x.max(),
                    "time": lambda x: x.mean(),
                })
                report = report.reset_index()
            report["dataset"] = "Housing"
            report["problem_type"] = "regression"
            reports.append(report)

            X, y = load_diabetes(return_X_y=True)
            report = benchmark(X, y, epsilon=epsilon, delta=1e-5, problem_type="regression")
            if not detailed:
                report = report.groupby(["model", "type", "epsilon"]).agg({
                    "accuracy": lambda x: x.max(),
                    "time": lambda x: x.mean(),
                })
                report = report.reset_index()
            report["dataset"] = "Diabetes"
            report["problem_type"] = "regression"
            reports.append(report)

    df = pd.concat(reports).reindex()
    df.to_csv("benchmark.csv", index=False)
    df = df.groupby(["model", "type", "epsilon", "dataset", "problem_type"]).agg({
        "accuracy": lambda x: x.mean(),
        "time": lambda x: x.mean(),
    })
    print(df)
