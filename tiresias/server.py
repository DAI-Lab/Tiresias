import uuid
import threading
import numpy as np
import tiresias.mechanisms as mechanisms
from time import sleep
from json import loads, dumps
from bottle import Bottle, request, response

def execute(query, payloads):
    if query["type"] == "basic":
        query["result"] = execute_basic(query, payloads)
        query["status"] = "COMPLETE"
    elif query["type"] == "generalized":
        query["result"] = execute_generalized(query, payloads)
        query["status"] = "COMPLETE"
    else:
        raise ValueError("Unknown query type: %s" % query["type"])

def execute_basic(query, payloads):
    if query["aggregator"] == "median":
        return mechanisms.dp_median(payloads, epsilon=query["epsilon"])
    elif query["aggregator"] == "mean":
        return mechanisms.dp_mean(payloads, epsilon=query["epsilon"])
    elif query["aggregator"] == "sum":
        return mechanisms.dp_sum(payloads, epsilon=query["epsilon"])
    else:
        raise ValueError("Unknown aggregator: %s" % query["aggregator"])

def execute_generalized(query, payloads):
    payloads = np.array(payloads)
    np.random.shuffle(payloads)
    x = []
    for partition in np.array_split(payloads, 5):
        inputs = partition.tolist()
        outputs = {}
        exec(query["sampler"], {}, {"inputs": inputs, "outputs": outputs})
        x.append(outputs["x"])
    if query["aggregator"] == "median":
        return mechanisms.dp_median(x, epsilon=query["epsilon"])
    else:
        raise ValueError("Unknown aggregator: %s" % query["aggregator"])

def validate_query(query):
    assert "epsilon" in query
    assert query["epsilon"] >= 0.0
    if query["type"] == "basic":
        assert "featurizer" in query
        assert query["aggregator"] in set(["median", "mean", "sum"])
    elif query["type"] == "generalized":
        assert "featurizer" in query
        assert "sampler" in query
        assert query["aggregator"] in set(["median"])
    else:
        raise ValueError("Unknown query type: %s" % query["type"])

def validate_payload(query, payload):
    if query["type"] == "basic":
        assert type(payload) == float
    elif query["type"] == "generalized":
        assert type(payload) == dict
    else:
        raise ValueError("Unknown query type: %s" % query["type"])

def run(port=3000):
    api = Bottle()
    queries = {}
    payloads = {}

    @api.route("/")
    def index():
        response.content_type = "application/json"
        return dumps(queries, indent=2)

    @api.route("/query")
    def create_query():
        query_id = str(uuid.uuid4())
        query = loads(request.params.get("query"))
        validate_query(query)
        query["id"] = query_id
        query["status"] = "PENDING"
        queries[query_id] = query
        payloads[query_id] = []
        return str(query_id)

    @api.route("/query/<query_id>")
    def fetch_query(query_id):
        return queries[query_id]

    @api.route("/query/<query_id>/submit")
    def approve_query(query_id):
        query = queries[query_id]
        payload = loads(request.params.get("payload"))
        validate_payload(query, payload)
        payloads[query_id].append(payload)
        query["count"] = len(payloads[query_id])
        query["status"] = "RUNNING"
        return ""

    api_thread = threading.Thread(target=api.run, kwargs={"port": port})
    api_thread.start()
    while api_thread.is_alive():
        for qid in queries.keys():
            try:
                print("Attempting to execute %s on %s" % (queries[qid], payloads[qid]))
                execute(queries[qid], payloads[qid])
            except:
                print("Query failed: %s" % queries[qid])
        sleep(10)
