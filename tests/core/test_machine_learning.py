import torch
import tiresias.core.machine_learning as ml

def test_naive_bayes():
    spec = {
        "model": "GaussianNB",
        "inputs": ["x0", "x1"],
        "output": "y",
        "bounds": [(0.0, 1.0), (0.0, 1.0)]
    }
    data = [
        {"x0": 0.0, 'x1': 1.0, "y": 1},
        {"x0": 0.0, 'x1': 0.0, "y": 0},
        {"x0": 1.0, 'x1': 1.0, "y": 1},
        {"x0": 1.0, 'x1': 0.0, "y": 0},
    ]
    model = ml.compute(spec, data, epsilon=100.0)
    assert int(model.predict([[0.5, 0.0]])[0]) == 0
    assert int(model.predict([[0.5, 1.0]])[0]) == 1

def test_logistic_regression():
    spec = {
        "model": "LogisticRegression",
        "inputs": ["x0", "x1"],
        "output": "y",
        "data_norm": 2.0
    }
    data = [
        {"x0": 0.0, 'x1': 1.0, "y": 1},
        {"x0": 0.0, 'x1': 0.0, "y": 0},
        {"x0": 1.0, 'x1': 1.0, "y": 1},
        {"x0": 1.0, 'x1': 0.0, "y": 0},
    ]
    model = ml.compute(spec, data, epsilon=100.0)
    assert int(model.predict([[0.5, 0.0]])[0]) == 0
    assert int(model.predict([[0.5, 1.0]])[0]) == 1
