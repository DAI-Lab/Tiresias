"""
This module provides helper functions for calling the REST API.
"""
import requests
import urllib.parse
from time import sleep
from json import loads, dumps
from tiresias.core import b64_decode

def retry(func):
    nb_tries = 3
    def _f(*args, **kwargs):
        for i in range(1, nb_tries + 1):
            try:
                results = func(*args, **kwargs)
                break
            except requests.exceptions.ConnectionError:
                sleep(0.1 * i)
                if i == nb_tries:
                    raise
        return results
    return _f

@retry
def list_tasks(server):
    """
    This helper function submits a GET request to obtain a list of queries.
    """
    return loads(requests.get(urllib.parse.urljoin(server, "/list")).text)

@retry
def create_task(server, task):
    """
    This helper function submits a GET request to create a new query. See the 
    client-side query handler `tiersias.client.handler` documentation for 
    examples of valid queries.
    """
    return requests.get(urllib.parse.urljoin(server, "/task"), params={
        "task": dumps(task)
    }).text

@retry
def approve_task(server, task_id, payload):
    """
    This helper function submits a GET request to contribute data to a given
    query.
    """
    return requests.get(urllib.parse.urljoin(server, "/task/%s/submit" % task_id), params={
        "payload": dumps(payload)
    })

@retry
def fetch_task(server, task_id):
    """
    This helper function submits a GET request to obtain a list of queries.
    """
    obj = requests.get(urllib.parse.urljoin(server, "/task/%s" % task_id)).text
    obj = loads(obj)
    if "error" in obj:
        raise ValueError(obj["error"])
    try:
        if "result" in obj:
            obj["result"] = b64_decode(obj["result"])
    except:
        pass
    return obj
