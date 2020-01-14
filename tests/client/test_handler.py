from tiresias.client import handler, storage
from tiresias.core import b64_encode, b64_decode

def test_handle_basic():
    task = {
        "type": "basic",
        "epsilon": 1.0,
        "featurizer": "SELECT * FROM dummy",
        "aggregator": "mean"
    }
    data = [{"x": 1.0}]
    result = handler.handle_basic(task, data)
    assert result[0] == 1.0

def test_handle_bounded():
    task = {
        "type": "bounded",
        "epsilon": 1.0,
        "featurizer": "SELECT * FROM dummy",
        "bounds": {
            "x": {"type": "set", "values": [0, 1, 2, 3], "default": 0},
            "y": {"type": "range", "low": 0.0, "high": 1.0},
        }
    }
    data = [{"x": 2, "y": 0.5}, {"x": 1, "y": 0.1}]
    result = handler.handle_bounded(task, data)
    for i in range(len(result)):
        assert result[i]["x"] in set([0, 1, 2, 3])
        assert type(result[i]["y"]) == float

def test_handle_gradient():
    import torch
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
    data = [
        {"x0": 0.0, "x1": 1.0, "y": 1.0},
        {"x0": 1.0, "x1": 1.0, "y": 2.0}
    ]
    result = handler.handle_gradient(task, data)
    assert b64_decode(result)

def test_handle_task(tmpdir):
    storage.initialize(tmpdir)
    storage.register_app(tmpdir, "example_app", {
        "tableA": {
            "description": "This table contains A.",
            "columns": {
                "some_var": {
                    "type": "float",
                    "description": "This column contains 1."
                }
            }
        }
    })
    storage.insert_payload(tmpdir, "example_app", {
        "tableA": [
            {"some_var": 1.0},
            {"some_var": 2.0},
            {"some_var": 3.0},
            {"some_var": 4.0},
            {"some_var": 5.0},
        ]
    })

    task = {
        "type": "basic",
        "epsilon": 1.0,
        "featurizer": "SELECT sum(some_var) FROM example_app.tableA",
        "aggregator": "mean"
    }
    result, err = handler.handle_task(tmpdir, task)
    assert result[0] == 15.0
