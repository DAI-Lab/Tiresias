import os
import requests
import threading
import urllib.parse
from time import sleep
from json import loads, dumps
from random import random, randint
from bottle import Bottle, request, response, static_file
import tiresias.server as server
import tiresias.server.remote
from tiresias.client.handler import handle_task
from tiresias.client.storage import execute_sql
from tiresias.client.storage import initialize, app_columns, register_app, insert_payload
from tiresias.client.synthetic import create_synthetic_dataset

def run(server_url, storage_dir, storage_port, policy, synthetic):
    whitelist = set()

    storage_thread = threading.Thread(target=storage_server, args=(storage_dir, storage_port, server_url, whitelist, synthetic))
    storage_thread.start()
    sleep(0.1)

    handler_thread = threading.Thread(target=task_handler, args=(server_url, storage_dir, whitelist, policy))
    handler_thread.start()
    sleep(0.1)

    storage_thread.join()
    handler_thread.join()

def storage_server(storage_dir, storage_port, server_url, whitelist, synthetic):
    api = Bottle()
    initialize(storage_dir)
    create_dummy_dataset(storage_dir)
    if synthetic:
        create_synthetic_dataset(storage_dir)
    api.config['storage_dir'] = storage_dir

    @api.route("/")
    def _index():
        root = os.path.dirname(__file__)
        return static_file('client.html', root=root)

    @api.route("/tasks")
    def _tasks():
        tasks = tiresias.server.remote.list_tasks(server_url)
        response.content_type = "application/json"
        for task_id, task in tasks.items():
            task["accepted"] = task_id in whitelist
            task["preview"] = execute_sql(storage_dir, task["featurizer"])
        return tasks

    @api.route("/whitelist/<task_id>")
    def _whitelist_task(task_id):
        whitelist.add(task_id)
        return ""

    @api.route("/app")
    def _app():
        """
        This REST endpoint returns a JSON array containing a list of the columns stored on the device.
        """
        rows = app_columns(api.config['storage_dir'])
        response.content_type = "application/json"
        return dumps(rows, indent=2)

    @api.route("/app/<app_name>/register")
    def _register(app_name):
        """
        This REST endpoint allows a new application to register by providing their database schema. The
        `schema` parameter is a JSON object.
        """
        schema = loads(request.params.get("schema"))
        register_app(api.config['storage_dir'], app_name, schema)
        return ""

    @api.route("/app/<app_name>/insert")
    def _insert(app_name):
        """
        This REST endpoint allows an application to append rows to their database by submitting a JSON 
        object in the `payload` field.
        """
        payload = loads(request.params.get("payload"))
        insert_payload(api.config['storage_dir'], app_name, payload)
        return ""

    api.run(host="localhost", port=storage_port, quiet=True)

def task_handler(server_url, storage_dir, whitelist, policy):
    processed = set()
    while True:
        try:
            tasks = server.remote.list_tasks(server_url)
            for id, task in tasks.items():
                if id in processed:
                    continue
                if id in whitelist or policy == "accept":
                    result, err = handle_task(storage_dir, task)
                    if not err:
                        server.remote.approve_task(server_url, id, result)
                    else:
                        print(err)
                    processed.add(id)
        except requests.exceptions.ConnectionError:
            print("The server at %s is offline; retrying in 1s." % server_url)
            sleep(1.0)
        sleep(0.5 + random())

def create_dummy_dataset(storage_dir):
    from sklearn.datasets import load_wine
    
    wine_dataset = load_wine()
    wine_dataset.feature_names = [k.replace("/", "_") for k in wine_dataset.feature_names]

    register_app(storage_dir, "dummy", {
        "wine": {
            "description": "Rows sampled from the Wine classification dataset.",
            "columns": {x: {"type": "float", "description": x} for x in wine_dataset.feature_names}
        }
    })

    i = randint(0, len(wine_dataset.data)-1)
    insert_payload(storage_dir, "dummy", {
        "wine": [
            {k: v for k, v in zip(wine_dataset.feature_names, wine_dataset.data[i])}
        ]
    })
