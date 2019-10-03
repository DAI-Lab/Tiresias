import requests
import urllib.parse
from time import sleep
from json import loads, dumps

def execute(port, sql):
    response = requests.get("http://localhost:%s/query" % port, params={
        "sql": sql
    })
    return loads(response.text)

def list_queries(server):
    response = requests.get(server)
    yield from loads(response.text).values()

def approve_query(server, query_id, value):
    response = requests.get(urllib.parse.urljoin(server, "/query/%s/submit" % query_id), params={
        "payload": dumps(value)
    })
    print(response)

def run(server="http://localhost:3000", datastore_port=8000):
    handled = set()
    while True:
        print("Fetching queries...")
        for query in list_queries(server):
            if query["id"] in handled:
                continue
            print("Executing %s" % query)
            rows = execute(datastore_port, query["featurizer"])
            if query['type'] == 'basic': # basic queries return a single number
                value = rows[0] # [{}] -> {}
                value = float(list(value.values())[0]) # {a: #} -> #
            if query['type'] == 'generalized': # generalized queries return a single dictionary
                value = rows[0] # [{}] -> {}
            approve_query(server, query["id"], value)
            handled.add(query["id"])
        sleep(10)
