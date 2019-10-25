import requests
import tiresias.client as client
import tiresias.client.api as api
import tiresias.client.storage as storage
import tiresias.client.handler as handler
import tiresias.core.federated_learning as fl
from tiresias.core import b64_encode, b64_decode

from json import dumps
from time import sleep
from multiprocessing import Process

def test_basic(tmpdir):
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

    query = {
        "type": "basic",
        "epsilon": 1.0,
        "featurizer": "SELECT sum(some_var) FROM example_app.tableA",
        "aggregator": "mean"
    }
    result = handler.handle(query, tmpdir)
    assert result == 15.0

def test_api_basic(tmpdir):
    storage_server = Process(target=client.storage_server, args=(tmpdir, 8000))
    storage_server.start()
    sleep(1)

    try:
        # Register a dummy app with some dummy data through the REST API
        api.register_app(8000, "example_app", {
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
        api.insert_payload(8000, "example_app", {
            "tableA": [
                {"some_var": 1.0},
                {"some_var": 2.0},
                {"some_var": 3.0},
                {"some_var": 4.0},
                {"some_var": 5.0},
            ]
        })

        # Make sure the data was inserted correctly
        query = {
            "type": "basic",
            "epsilon": 1.0,
            "featurizer": "SELECT avg(some_var) FROM example_app.tableA",
            "aggregator": "mean"
        }
        result = handler.handle(query, tmpdir)
        assert result == 3.0
    
    finally:
        storage_server.terminate()

def test_api_federated_learning(tmpdir):
    storage_server = Process(target=client.storage_server, args=(tmpdir, 8000))
    storage_server.start()
    sleep(1)

    try:
        # Register a dummy app with some dummy data through the REST API
        api.register_app(8000, "example_app", {
            "tableA": {
                "description": "This table contains A.",
                "columns": {
                    "x1": {"type": "float", "description": ""},
                    "x2": {"type": "float", "description": ""},
                    "y": {"type": "float", "description": ""},
                }
            }
        })
        api.insert_payload(8000, "example_app", {
            "tableA": [
                {"x1": 0.0, "x2": 1.0, "y": 1.0}, # y = x1 + x2
                {"x1": 0.0, "x2": 2.0, "y": 2.0},
                {"x1": 1.0, "x2": 3.0, "y": 4.0},
                {"x1": 1.0, "x2": 4.0, "y": 5.0},
            ]
        })

        # Make sure the data was inserted correctly
        query = {
            "type": "federated_learning",
            "epsilon": 10.0,
            "featurizer": "SELECT x1, x2, y FROM example_app.tableA",
            "aggregator": {
                "lr": 1e-4,
                "model": "Linear",
                "inputs": ["x1", "x2"],
                "outputs": ["y"],
                "loss": "MSE"
            }
        }
        query["weights"] = b64_encode(fl.random_weights(query["aggregator"]))
        gradients = b64_decode(handler.handle(query, tmpdir))
        assert gradients[0].size(1) == 2, "The gradients for w1*x1 + w2*x2."
        assert gradients[1].size(0) == 1, "The gradient for the bias terrm."
    
    finally:
        storage_server.terminate()
