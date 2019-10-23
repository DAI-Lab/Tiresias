from time import sleep
from random import random
from multiprocessing import Process
import tiresias.server as server

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
