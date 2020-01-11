"""
This module provides functions for handling different types of queries.
"""
from tiresias.core import b64_encode, b64_decode
from tiresias.client.storage import execute_sql
from tiresias.core.federated_learning import gradients
from tiresias.core.mechanisms import finite_categorical, bounded_continuous

def handle(query, data_dir):
    """
    This function takes a generic query and dispatches it to the correct query
    handler. It's also responsbile for extracting the correct features from the 
    given data directory.
    """
    try:
        dispatcher = {
            "basic": handle_basic,
            "bounded": handle_bounded,
            "machine_learning": handle_ml,
            "federated_learning": handle_fl
        }
        func = dispatcher.get(query["type"], lambda: ValueError("Unknown query type."))
        data = execute_sql(data_dir, query["featurizer"])
        return func(query, data)
    except:
        print("Query failed.")
        return None

def handle_basic(query, data):
    """
    This function handles "basic" queries. The `featurizer` is expected to 
    return a single scalar value which is not differentially private. On the 
    server side, these scalar values will be aggregated using the provided
    aggregation function to produce a differentially private summary statistic.
    
    An example of a basic query is:

    ```
    {
        "type": "basic",
        "epsilon": 1.0,
        "featurizer": "SELECT age FROM user.profile",
        "aggregator": "mean"
    }
    ```
    """
    assert type(data) == list, "Featurizers should return rows."
    assert len(data) == 1, "Featurizers for basic queries should return a single row."
    assert type(data[0]) == dict, "Featurizers should return rows of dictionaries."
    assert len(data[0]) == 1, "Featurizers for basic queries should return a row with a single value"

    value = list(data[0].values())[0]
    return value

def handle_bounded(query, data):
    """
    This function handles "bounded" queries. The `featurizer` is expected to 
    return a dictionary of key-value pairs where the values are bounded. The
    bounds for each variable must be specified as either a set of discrete 
    values or a range of real values.

    Local differential privacy mechanisms are used to anonymize this set of
    values before they are transmitted. An example of a bounded query is:

    ```
    {
        "type": "bounded",
        "epsilon": 1.0,
        "featurizer": "SELECT species, weight FROM user.pets",
        "bounds": {
            "species": {
                "type": "set", 
                "default": "dog",
                "values": ["cat", "dog"], 
            },
            "weight": {"type": "range", "low": 0.0, "high": 100.0},
        }
    }
    ```
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
            print(value, domain)
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
    This function handles "machine learning" queries. The `featurizer` is 
    expected to return a dictionary of key-value pairs where the values are 
    scalars. In addition, the `aggregator` must correspond to a differentially
    private machine learning technique which is implemented under 
    `tiresias.core.machine_learning`.

    An example of a machine learning query is:
    ```
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
    ```
    """
    assert type(data) == list, "Featurizers should return rows."
    assert len(data) == 1, "Featurizers for basic queries should return a single row."
    assert type(data[0]) == dict, "Featurizers should return rows of dictionaries."
    return data[0]

def handle_fl(query, data):
    """
    This function handles "federated learning" queries. The `featurizer` is 
    expected to return a list containing dictionaries of key-value pairs. The 
    `aggregator` must be a valid specification of a gradient-based model as 
    described in `tiresias.core.federated_learning`.

    The gradients are computed on the client side using their data and only 
    anonymized gradients are sent to the server. Each federated learning query
    corresponds to a single epoch in the training process. An example of a 
    federated learning query is:
    ```
    {
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
    ```
    """
    assert type(data) == list, "Featurizers should return rows."
    assert type(data[0]) == dict, "Featurizers should return rows of dictionaries."
    return b64_encode(gradients(query["aggregator"], b64_decode(query["weights"]), data, query["epsilon"]))
