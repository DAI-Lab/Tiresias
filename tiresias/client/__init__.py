# pylint: disable=no-member
import requests
import threading
import urllib.parse
from time import sleep
from json import loads, dumps
import tiresias.server as remote
from tiresias.client.handler import handle
from tiresias.client.storage import initialize, app_columns, execute_sql, register_app, insert_payload

def run(server="http://localhost:3000", data_dir="/tmp/tiresias", port=8000):
    storage_thread = threading.Thread(target=storage_server, args=(data_dir, port))
    storage_thread.start()

    query_thread = threading.Thread(target=query_handler, args=(server, data_dir))
    query_thread.start()

    storage_thread.join()
    query_thread.join()

def storage_server(data_dir="/tmp/tiresias", port=8000):
    from bottle import Bottle, request, response
    
    api = Bottle()
    initialize(data_dir)
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
    handled = set()
    while True:
        try:
            for query_id, query in remote.api.list_queries(server).items():
                if query_id in handled:
                    continue
                result = handle(query, data_dir)
                remote.api.approve_query(server, query_id, result)
                handled.add(query_id)
        except requests.exceptions.ConnectionError:
            print("Server at %s is offline." % server)
        sleep(0.5)
