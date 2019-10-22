from tiresias.client.storage import execute_sql
from tiresias.core.federated_learning import compute
from tiresias.core.mechanisms import finite_categorical, bounded_continuous

def handle(query, data_dir):
    dispatcher = {
        "basic": handle_basic,
        "bounded": handle_bounded,
        "machine_learning": handle_ml,
        "federated_learning": handle_fl
    }
    func = dispatcher.get(query["type"], lambda: ValueError("Unknown query type."))
    data = execute_sql(data_dir, query["featurizer"])
    return func(query, data)

def handle_basic(query, data):
    """
    {
        "type": "basic",
        "epsilon": 1.0,
        "featurizer": "<SQL>",
        "aggregator": "mean"
    }
    """
    assert type(data) == list, "Featurizers should return rows."
    assert len(data) == 1, "Featurizers for basic queries should return a single row."
    assert type(data[0]) == dict, "Featurizers should return rows of dictionaries."
    assert len(data[0]) == 1, "Featurizers for basic queries should return a row with a single value"

    value = list(data[0].values())[0]
    return value

def handle_bounded(query, data):
    """
    {
        "type": "bounded",
        "epsilon": 1.0,
        "featurizer": "<SQL>",
        "bounds": {
            "<var_name_1>": {"type": "set", "values": [1, 2, 3], "default": 0},
            "<var_name_2>": {"type": "range", "low": 0.0, "high": 1.0},
        }
    }
    """
    assert type(data) == list, "Featurizers should return rows."
    assert len(data) == 1, "Featurizers for basic queries should return a single row."
    assert type(data[0]) == dict, "Featurizers should return rows of dictionaries."

    def ldp(value, bounds, epsilon):
        if bounds["type"] == "range":
            value = min(max(bounds["low"], value), bounds["high"])
            return bounded_continuous(value, bounds["low"], bounds["high"], epsilon)
        elif bounds["type"] == "set":
            if value not in bounds["values"]:
                value = bounds["default"]
            domain = set(bounds["values"])
            domain.add(bounds["default"])
            return finite_categorical(value, domain, epsilon)
        return ValueError("Unknown bounds.")

    result = {}
    for key, value in data[0].items():
        assert key in query["bounds"]
        bounds = query["bounds"][key]
        result[key] = ldp(value, bounds, query["epsilon"])
    return result

def handle_ml(query, data):
    """
    {
        "type": "machine_learning",
        "epsilon": 1.0,
        "featurizer": "<SQL>",
        "aggregator": {
            "model": "LogisticRegression",
            "inputs": ["<var_name>", "<var_name>"],
            "outputs": ["<var_name>""],
        }
    }
    """
    assert type(data) == list, "Featurizers should return rows."
    assert len(data) == 1, "Featurizers for basic queries should return a single row."
    assert type(data[0]) == dict, "Featurizers should return rows of dictionaries."
    return data[0]

def handle_fl(query, data):
    """
    {
        "type": "federated_learning",
        "epsilon": 1.0,
        "featurizer": "<SQL>",
        "aggregator": {
            "model": "MultilayerPerceptron",
            "inputs": ["<var_name>", "<var_name>"],
            "outputs": ["<var_name>"],
        },
        "weights": "<binary_blob>"
    }
    """
    assert type(data) == list, "Featurizers should return rows."
    assert len(data) == 1, "Featurizers for basic queries should return a single row."
    assert type(data[0]) == dict, "Featurizers should return rows of dictionaries."
    return compute(data["aggregator"], data["weights"], data[0])
