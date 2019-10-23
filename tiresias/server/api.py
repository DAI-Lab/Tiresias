import requests
import urllib.parse
from json import loads, dumps

def list_queries(server):
    return loads(requests.get(server).text)

def create_query(server, query):
    return requests.get(urllib.parse.urljoin(server, "/query"), params={
        "query": dumps(query)
    }).text

def approve_query(server, query_id, payload):
    return requests.get(urllib.parse.urljoin(server, "/query/%s/submit" % query_id), params={
        "payload": dumps(payload)
    })
