from time import sleep
from random import random
from multiprocessing import Process
from tiresias.core import b64_encode, b64_decode

import torch
import tiresias.server as server
import tiresias.server.api
import tiresias.client as client
import tiresias.client.api
import tiresias.core.federated_learning as fl

def test_create_query():
    api_server = Process(target=server.run, args=(3000,))
    api_server.start()
    sleep(1)

    try:
        queries = server.api.list_queries("http://localhost:3000/")
        assert len(queries) == 0

        query_id = server.api.create_query("http://localhost:3000/", {
            "type": "basic",
            "epsilon": 1.0,
            "featurizer": "SELECT sum(some_var) FROM example_app.tableA",
            "aggregator": "mean"
        })
        queries = server.api.list_queries("http://localhost:3000/")
        assert len(queries) == 1
        assert queries[query_id]["id"] == query_id
        assert queries[query_id]["type"] == "basic"

    finally:
        api_server.terminate()

def test_approve_query():
    api_server = Process(target=server.run, args=(3000,))
    api_server.start()
    sleep(1)

    try:
        query_id = server.api.create_query("http://localhost:3000/", {
            "type": "basic",
            "epsilon": 100.0,
            "featurizer": "SELECT sum(some_var) FROM example_app.tableA",
            "aggregator": "mean"
        })
        queries = server.api.list_queries("http://localhost:3000/")
        assert len(queries) == 1
        assert queries[query_id]["id"] == query_id
        assert queries[query_id]["type"] == "basic"

        for _ in range(10):
             server.api.approve_query("http://localhost:3000/", query_id, 100.0 + random())
        sleep(1)

        queries = server.api.list_queries("http://localhost:3000/")
        assert len(queries) == 1
        assert queries[query_id]["id"] == query_id
        assert abs(queries[query_id]["result"] - 100.0) < 10.0

    finally:
        api_server.terminate()

def test_server_client_basic(tmpdir):
    api_server = Process(target=server.run, args=(3000,))
    api_server.start()
    sleep(1)

    client_server = Process(target=client.run, args=("http://localhost:3000", tmpdir, 8000))
    client_server.start()
    sleep(1)

    try:
        # Register a dummy app with some dummy data through the REST API
        client.api.register_app(8000, "example_app", {
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
        client.api.insert_payload(8000, "example_app", {
            "tableA": [
                {"some_var": 1.0},
                {"some_var": 2.0},
                {"some_var": 3.0},
                {"some_var": 4.0},
                {"some_var": 5.0},
            ]
        })

        # Submit a query
        query_id = server.api.create_query("http://localhost:3000/", {
            "type": "basic",
            "epsilon": 100.0,
            "featurizer": "SELECT sum(some_var) FROM example_app.tableA",
            "aggregator": "mean"
        })
        sleep(1)

        # Check that the client responded to the query
        queries = server.api.list_queries("http://localhost:3000/")
        assert len(queries) == 1
        assert queries[query_id]["id"] == query_id
        assert queries[query_id]["count"] == 1 # One data point

    finally:
        client_server.terminate()
        api_server.terminate()

def test_server_client_with_values(tmpdir):
    api_server = Process(target=server.run, args=(3000,))
    api_server.start()
    sleep(1)

    client_server = Process(target=client.run, args=("http://localhost:3000", tmpdir, 8000))
    client_server.start()
    sleep(1)

    try:
        # Register a dummy app with some dummy data through the REST API
        client.api.register_app(8000, "example_app", {
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
        client.api.insert_payload(8000, "example_app", {
            "tableA": [
                {"some_var": 100.0},
                {"some_var": 110.0},
                {"some_var": 90.0}
            ]
        })

        # Submit a query
        query_id = server.api.create_query("http://localhost:3000/", {
            "type": "basic",
            "epsilon": 10000.0,
            "featurizer": "SELECT avg(some_var) FROM example_app.tableA",
            "aggregator": "mean"
        })
        sleep(1)

        # Submit some dummy data directly without using the client
        for _ in range(9):
             server.api.approve_query("http://localhost:3000/", query_id, 100.0 + random())
        sleep(1)

        # Check that the client responded to the query
        queries = server.api.list_queries("http://localhost:3000/")
        assert len(queries) == 1
        assert queries[query_id]["id"] == query_id
        assert queries[query_id]["count"] == 10 # One "real" point + 9 patched ones
        assert abs(queries[query_id]["result"] - 100.0) < 1.0 # One data point

        # Check that the client responded to the query (direct)
        query = server.api.fetch_query("http://localhost:3000/", query_id)
        assert query["count"] == 10 # One "real" point + 9 patched ones
        assert abs(query["result"] - 100.0) < 1.0 # One data point

    finally:
        client_server.terminate()
        api_server.terminate()

def test_server_client_with_fl(tmpdir):
    api_server = Process(target=server.run, args=(3000,))
    api_server.start()
    sleep(1)

    client_server = Process(target=client.run, args=("http://localhost:3000", tmpdir, 8000))
    client_server.start()
    sleep(1)

    try:
        # Register a dummy app with some dummy data through the REST API
        client.api.register_app(8000, "example_app", {
            "tableA": {
                "description": "This table contains A.",
                "columns": {
                    "x1": {"type": "float", "description": ""},
                    "x2": {"type": "float", "description": ""},
                    "y": {"type": "float", "description": ""},
                }
            }
        })
        client.api.insert_payload(8000, "example_app", {
            "tableA": [
                {"x1": 0.0, "x2": 1.0, "y": 1.0}, # y = x1 + x2
                {"x1": 0.0, "x2": 2.0, "y": 2.0},
                {"x1": 1.0, "x2": 3.0, "y": 4.0},
                {"x1": 1.0, "x2": 4.0, "y": 5.0},
            ]
        })

        # Submit a query
        query = {
            "type": "federated_learning",
            "epsilon": 10.0,
            "featurizer": "SELECT x1, x2, y FROM example_app.tableA",
            "aggregator": {
                "lr": 1e-4,
                "model": "Linear",
                "inputs": ["x1", "x2"],
                "outputs": ["y"],
                "loss": "MSE",
            }
        }
        query["weights"] = b64_encode(fl.random_weights(query["aggregator"]))
        query_id = server.api.create_query("http://localhost:3000/", query)
        sleep(2)

        # Check that the client responded to the query
        queries = server.api.list_queries("http://localhost:3000/")
        assert len(queries) == 1
        assert queries[query_id]["id"] == query_id
        assert queries[query_id]["count"] == 1 # One "real" point

        model = b64_decode(queries[query_id]["result"])
        assert type(model) == type(torch.nn.Linear(1, 1))

    finally:
        client_server.terminate()
        api_server.terminate()
