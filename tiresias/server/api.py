"""
This module provides helper functions for calling the REST API.
"""
import requests
import urllib.parse
from json import loads, dumps

def list_queries(server):
    """
    This helper function submits a GET request to obtain a list of queries.
    """
    return loads(requests.get(server).text)

def create_query(server, query):
    """
    This helper function submits a GET request to create a new query. See the 
    client-side query handler `tiersias.client.handler` documentation for 
    examples of valid queries.
    """
    return requests.get(urllib.parse.urljoin(server, "/query"), params={
        "query": dumps(query)
    }).text

def approve_query(server, query_id, payload):
    """
    This helper function submits a GET request to contribute data to a given
    query.
    """
    return requests.get(urllib.parse.urljoin(server, "/query/%s/submit" % query_id), params={
        "payload": dumps(payload)
    })

def fetch_query(server, query_id):
    """
    This helper function submits a GET request to obtain a list of queries.
    """
    return loads(requests.get(urllib.parse.urljoin(server, "/query/%s" % query_id)).text)
