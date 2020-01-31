from time import sleep
from random import randint
from multiprocessing import Process
import tiresias.client as client
import tiresias.client.remote as remote
import tiresias.client.handler as handler

TEST_PORT = 8000 + randint(0, 1000)

def test_storage_server(tmpdir):
    storage_server = Process(target=client.storage_server, args=(tmpdir, TEST_PORT, "", set(), False))
    storage_server.start()
    sleep(0.1)

    try:
        # Register a dummy app with some dummy data through the REST API
        remote.register_app(TEST_PORT, "example_app", {
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
        remote.insert_payload(TEST_PORT, "example_app", {
            "tableA": [
                {"some_var": 1.0},
                {"some_var": 2.0},
                {"some_var": 3.0},
                {"some_var": 4.0},
                {"some_var": 5.0},
            ]
        })

        # Make sure the data was inserted correctly
        task = {
            "type": "basic",
            "epsilon": 1.0,
            "featurizer": "SELECT some_var FROM example_app.tableA",
            "aggregator": "mean"
        }
        result, err = handler.handle_task(tmpdir, task)
        assert result[0] == 1.0
        assert result[1] == 2.0
    
    finally:
        storage_server.terminate()
