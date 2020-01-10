"""
The `tiresias.core.machine_learning` module provides implementations of 
differentially private machine learning techniques.
"""
import numpy as np
import diffprivlib.models as dp

GaussianNB = dp.GaussianNB
LogisticRegression = dp.LogisticRegression
LinearRegression = dp.LinearRegression

def compute(spec, data, epsilon):
    """
    Given the model specification and some data, train a differeentially 
    private model.
    """
    x = np.array([[d[var] for var in spec["inputs"]] for d in data])
    y = np.array([d[spec["output"]] for d in data])
    if spec["model"] == "GaussianNB":
        clf = GaussianNB(epsilon=epsilon, bounds=spec["bounds"])
        clf.fit(x, y)
        return clf
    elif spec["model"] == "LogisticRegression":
        clf = LogisticRegression(epsilon=epsilon, data_norm=spec["data_norm"])
        clf.fit(x, y)
        return clf
    elif spec["model"] == "LinearRegression":
        clf = LinearRegression(epsilon=epsilon)
        clf.fit(x, y)
        return clf
    else:
        raise ValueError(spec["model"])
