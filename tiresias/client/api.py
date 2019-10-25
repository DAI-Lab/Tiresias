import requests
from json import loads, dumps
from .storage import validate_schema, validate_payload

def register_app(port, app_name, schema):
    validate_schema(schema)
    return requests.get("http://localhost:%s/app/example_app/register" % port, params={"schema": dumps(schema)})

def insert_payload(port, app_name, payload):
    validate_payload(payload)
    return requests.get("http://localhost:%s/app/example_app/insert" % port, params={"payload": dumps(payload)})
