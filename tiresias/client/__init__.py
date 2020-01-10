"""
The `tiresias.client` module is responsible for (1) providing a local personal 
data store and (2) performing computations necessary for differential privacy 
upon request. At the top-level, this module provides functions for (1) running 
a storage server which provides a REST API for applications to store data in
the personal data store and (2) starting a thread which periodically looks for 
queries that the user can contribute to and performs the computations that are
needed to securely contribute to them.
"""
# pylint: disable=no-member
import requests
import threading
import urllib.parse
from time import sleep
from json import loads, dumps
from tiresias.server import api as remote_api
from tiresias.client.handler import handle
from tiresias.client.storage import initialize, create_dummy_dataset, app_columns, execute_sql, register_app, insert_payload

def run(server="http://localhost:3000", data_dir="/tmp/tiresias", port=8000):
    """
    This function launches the storage server and query handler on different 
    threads and blocks forever.
    """
    storage_thread = threading.Thread(target=storage_server, args=(data_dir, port))
    storage_thread.start()

    query_thread = threading.Thread(target=query_handler, args=(server, data_dir))
    query_thread.start()

    storage_thread.join()
    query_thread.join()

def storage_server(data_dir="/tmp/tiresias", port=8000):
    """
    This function launches the storage server. It stores the SQLite databases 
    in the given data directory and listens on the given port. Helper functions
    for interacting with this API are provided under `tiresias.client.api`.
    """
    from bottle import Bottle, request, response
    
    api = Bottle()
    initialize(data_dir)
    create_dummy_dataset(data_dir)
    api.config['data_dir'] = data_dir

    @api.route("/")
    def _index():
        """
        This REST endpoint returns a JSON array containing a list of the columns stored on the device.
        """
        rows = app_columns(api.config['data_dir'])
        response.content_type = "application/json"
        return dumps(rows, indent=2)

    @api.route("/query")
    def _query():
        """
        This REST endpoint accepts a `sql` parameter which contains the SQL query. It attaches all the
        application databases to the primary metadata database and executes the query. Note that this 
        endpoint does not perform any security checks.
        """
        rows = execute_sql(api.config['data_dir'], request.params.get("sql"))
        response.content_type = "application/json"
        return dumps(rows, indent=2)

    @api.route("/app/<app_name>/register")
    def _register(app_name):
        """
        This REST endpoint allows a new application to register by providing their database schema. The
        `schema` parameter is a JSON object.
        """
        schema = loads(request.params.get("schema"))
        register_app(api.config['data_dir'], app_name, schema)
        return ""

    @api.route("/app/<app_name>/insert")
    def _insert(app_name):
        """
        This REST endpoint allows an application to append rows to their database by submitting a JSON 
        object in the `payload` field.
        """
        payload = loads(request.params.get("payload"))
        insert_payload(api.config['data_dir'], app_name, payload)
        return ""

    api.run(host="localhost", port=port)

def query_handler(server, data_dir):
    """
    This function launches the query handler. It repeatedly asks the server for
    new queries to process and then processes them.
    """
    handled = set()
    while True:
        try:
            for query_id, query in remote_api.list_queries(server).items():
                if query_id in handled:
                    continue
                result = handle(query, data_dir)
                remote_api.approve_query(server, query_id, result)
                handled.add(query_id)
        except requests.exceptions.ConnectionError:
            print("Server at %s is offline." % server)
        sleep(1.0)
