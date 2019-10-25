# pylint: disable=no-member
import uuid
import threading
import numpy as np
import tiresias.worker as worker
from time import sleep
from json import loads, dumps

def run(port=3000):
    from bottle import Bottle, request, response

    api = Bottle()
    queries = {}
    payloads = {}

    @api.route("/")
    def _index():
        response.content_type = "application/json"
        return dumps(queries, indent=2)

    @api.route("/query")
    def _create_query():
        query_id = str(uuid.uuid4())
        query = loads(request.params.get("query"))
        query["id"] = query_id
        query["status"] = "PENDING"
        queries[query_id] = query
        payloads[query_id] = []
        return str(query_id)

    @api.route("/query/<query_id>")
    def _fetch_query(query_id):
        return queries[query_id]

    @api.route("/query/<query_id>/submit")
    def _approve_query(query_id):
        query = queries[query_id]
        payload = loads(request.params.get("payload"))
        payloads[query_id].append(payload)
        query["count"] = len(payloads[query_id])
        query["status"] = "RUNNING"
        return ""

    api_thread = threading.Thread(target=api.run, kwargs={"port": port})
    api_thread.start()
    while api_thread.is_alive():
        for qid in queries.keys():
            try:
                queries[qid]["result"] = worker.handle(queries[qid], payloads[qid])
                queries[qid]["status"] = "COMPLETE"
            except:
                print("Query failed: %s" % queries[qid])
        sleep(0.1)
