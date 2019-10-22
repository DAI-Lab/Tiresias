import tiresias.client.handler as handler

def test_handle_basic():
    query = {
        "type": "basic",
        "epsilon": 1.0,
        "featurizer": "SELECT * FROM dummy",
        "aggregator": "mean"
    }
    data = [{"x": 1.0}]
    result = handler.handle_basic(query, data)
    assert result == 1.0

def test_handle_bounded():
    query = {
        "type": "bounded",
        "epsilon": 1.0,
        "featurizer": "SELECT * FROM dummy",
        "bounds": {
            "x": {"type": "set", "values": [0, 1, 2, 3], "default": 0},
            "y": {"type": "range", "low": 0.0, "high": 1.0},
        }
    }
    data = [{"x": 2, "y": 0.5}]
    result = handler.handle_bounded(query, data)
    assert result["x"] in set([0, 1, 2, 3])
    assert type(result["y"]) == float
