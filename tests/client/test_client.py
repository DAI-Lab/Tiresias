import requests
import tiresias.client as client
import tiresias.client.storage as storage
import tiresias.client.handler as handler
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

def test_remote_basic(tmpdir):
    storage_server = Process(target=client.storage_server, args=(tmpdir, 8000))
    storage_server.start()
    sleep(1)

    try:
        # Register a dummy app with some dummy data    
        requests.get("http://localhost:8000/app/example_app/register", params={"schema": dumps({
            "tableA": {
                "description": "This table contains A.",
                "columns": {
                    "some_var": {
                        "type": "float",
                        "description": "This column contains 1."
                    }
                }
            }
        })})
        requests.get("http://localhost:8000/app/example_app/insert", params={"payload": dumps({
            "tableA": [
                {"some_var": 1.0},
                {"some_var": 2.0},
                {"some_var": 3.0},
                {"some_var": 4.0},
                {"some_var": 5.0},
            ]
        })})

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
