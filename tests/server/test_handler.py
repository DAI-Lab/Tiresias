import torch
from random import random
from pytest import approx
import tiresias.client as client
from tiresias.server import handler
from tiresias.core import b64_encode, b64_decode

def test_handle_basic():
    task = {
        "type": "basic",
        "epsilon": 16.0,
        "delta": 1e-5,
        "featurizer": "SELECT * FROM dummy",
        "aggregator": "mean"
    }
    data = [[random()] for _ in range(100)]
    result = handler.handle_basic(task, data)
    assert result == approx(0.5, abs=0.1)

def test_handle_integrated():
    task = {
        "type": "integrated",
        "epsilon": 10.0,
        "featurizer": "SELECT x1, x2, y FROM profile.example",
        "model": "LinearRegression",
        "inputs": ["x0", "x1"],
        "output": "y"
    }
    data = [[{
        "x0": random(),
        "x1": random(),
        "y": random(),
    }] for _ in range(100)]
    result = handler.handle_integrated(task, data)
    assert result.predict

def test_handle_gradient():
    task = {
        "type": "gradient",
        "epsilon": 10.0,
        "delta": 1e-5,
        "lr": 0.01,
        "featurizer": "SELECT x1, x2, y FROM profile.example",
        "model": b64_encode(torch.nn.Sequential(
            torch.nn.Linear(2, 1)
        )),
        "loss": b64_encode(torch.nn.functional.mse_loss),
        "inputs": ["x0", "x1"],
        "output": ["y"],
    }

    gradients = []
    data = [
        {"x0": 0.0, "x1": 1.0, "y": 1.0},
        {"x0": 1.0, "x1": 1.0, "y": 2.0}
    ]
    for _ in range(10):
        gradients.append(client.handler.handle_gradient(task, data))

    result = handler.handle_gradient(task, gradients)
    assert b64_decode(result)
