"""
The `tiresias.server` module is responsible for providing a lightweight REST
API for creating queries and submitting data to queries. In addition, the 
server module also launches a `tiresias.worker` thread which is responsible 
for performing the differential privacy computations needed to produce the
final result for a given query.
"""
# pylint: disable=no-member
import uuid
import threading
import numpy as np
import tiresias.worker as worker
from time import sleep
from time import time
from json import loads, dumps

def run(port=3000):
    """
    This function launches a lightweight REST API for managing queries. In 
    addition, it spawns a single background `tiresias.worker` thread which 
    handles all incoming queries.
    """
    from bottle import Bottle, request, response

    api = Bottle()
    queries = {}
    payloads = {}
    memory_lock = threading.Lock()

    @api.route("/")
    def _index():
        with memory_lock:
            response.content_type = "application/json"
            return dumps(queries, indent=2)

    @api.route("/list")
    def _list():
        with memory_lock:
            response.content_type = "application/json"
            open_queries = {k: v for k, v in queries.items() if v["status"] == "PENDING"}
        return dumps(open_queries, indent=2)

    @api.route("/query")
    def _create_query():
        with memory_lock:
            query_id = str(uuid.uuid4())
            query = loads(request.params.get("query"))
            query["id"] = query_id
            query["status"] = "PENDING"
            query["start"] = time()
            queries[query_id] = query
            payloads[query_id] = []
            return str(query_id)

    @api.route("/query/<query_id>")
    def _fetch_query(query_id):
        with memory_lock:
            if query_id not in queries:
                return {"error": "Query not found."}
            return queries[query_id]

    @api.route("/query/<query_id>/submit")
    def _approve_query(query_id):
        with memory_lock:
            query = queries[query_id]
            if query["status"] != "PENDING":
                return ""
            payload = loads(request.params.get("payload"))
            payloads[query_id].append(payload)
            query["count"] = len(payloads[query_id])
            if "min_sample_size" not in query or query["count"] >= query["min_sample_size"]:
                query["status"] = "RUNNING"
            return ""

    api_thread = threading.Thread(target=api.run, kwargs={"port": port, "server": "paste"})
    api_thread.start()
    while api_thread.is_alive():
        # Remove expired queries
        with memory_lock:
            for qid in set(queries.keys()):
                if queries[qid]["status"] == "COMPLETE":
                    if time() - queries[qid]["start"] > 30:
                        del queries[qid]
        
        with memory_lock:
            for qid in queries.keys():
                if queries[qid]["status"] != "RUNNING":
                    continue
                try:
                    queries[qid]["result"] = worker.handle(queries[qid], payloads[qid])
                    queries[qid]["status"] = "COMPLETE"
                    queries[qid]["end"] = time()
                except Exception as e:
                    print("Query failed: %s" % queries[qid])
                    print(e)
        sleep(0.1)
