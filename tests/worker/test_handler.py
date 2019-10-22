import tiresias.worker.handler as handler

def test_handle_basic():
    query = {
        "type": "basic",
        "epsilon": 100.0,
        "featurizer": "SELECT * FROM dummy",
        "aggregator": "mean"
    }
    data = [1.0, 2.0, 3.0] * 100
    result = handler.handle_basic(query, data)
    assert abs(result - 2.0) < 1.0
