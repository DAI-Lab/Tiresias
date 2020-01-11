"""
This module provides helper functions for calling the REST API.
"""
import requests
import urllib.parse
from time import sleep
from json import loads, dumps

def try_three_times(func):
    def try_except_function(*args, **kwargs):
        """Modified version of func - see docstring for try_except().
        """
        for i in range(3):
            try:
                results = func(*args, **kwargs)
                break
            except requests.exceptions.ConnectionError:
                sleep(0.1)
                pass
        return results
    return try_except_function

@try_three_times
def list_queries(server):
    """
    This helper function submits a GET request to obtain a list of queries.
    """
    return loads(requests.get(urllib.parse.urljoin(server, "/list")).text)

@try_three_times
def create_query(server, query):
    """
    This helper function submits a GET request to create a new query. See the 
    client-side query handler `tiersias.client.handler` documentation for 
    examples of valid queries.
    """
    return requests.get(urllib.parse.urljoin(server, "/query"), params={
        "query": dumps(query)
    }).text

@try_three_times
def approve_query(server, query_id, payload):
    """
    This helper function submits a GET request to contribute data to a given
    query.
    """
    return requests.get(urllib.parse.urljoin(server, "/query/%s/submit" % query_id), params={
        "payload": dumps(payload)
    })

@try_three_times
def fetch_query(server, query_id):
    """
    This helper function submits a GET request to obtain a list of queries.
    """
    obj = loads(requests.get(urllib.parse.urljoin(server, "/query/%s" % query_id)).text)
    if "error" in obj:
        raise ValueError(obj["error"])
    return obj
