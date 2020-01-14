import os
import threading
import numpy as np
from time import sleep
from json import loads, dumps
from bottle import Bottle, request, response, static_file
from tiresias.server.platform import Platform

def run(port=3000):
    api = Bottle()
    platform = Platform()

    @api.route("/")
    def _index():
        root = os.path.dirname(__file__)
        return static_file('platform.html', root=root)

    @api.route("/all")
    def _all():
        response.content_type = "application/json"
        return dumps(platform.tasks(), indent=2)

    @api.route("/list")
    def _list():
        response.content_type = "application/json"
        return dumps(platform.tasks(only_pending=True), indent=2)

    @api.route("/task")
    def _create_task():
        task = loads(request.params.get("task"))
        return platform.create(task)

    @api.route("/task/<task_id>")
    def _fetch_task(task_id):
        return platform.fetch(task_id)

    @api.route("/task/<task_id>/submit")
    def _approve_task(task_id):
        payload = loads(request.params.get("payload"))
        return platform.submit(task_id, payload)

    api_thread = threading.Thread(target=api.run, kwargs={"port": port, "server": "paste", "host": "0.0.0.0"})
    api_thread.start()
    while api_thread.is_alive():
        platform.gc()
        platform.run()
        sleep(0.5)
