"""
This module provides helper functions for calling the REST API.
"""
import requests
from json import loads, dumps
from .storage import validate_schema, validate_payload

def register_app(port, app_name, schema):
    """
    This helper function submits a GET request to register a new application 
    with the storage server. See `tiresias.client.storage.register_app` for
    the backend implementation.
    """
    validate_schema(schema)
    return requests.get("http://localhost:%s/app/example_app/register" % port, params={"schema": dumps(schema)})

def insert_payload(port, app_name, payload):
    """
    This helper function submits a GET request to insert data collected by an
    existing application into the storage server. See 
    `tiresias.client.storage.insert_payload` for the backend implementation.
    """
    validate_payload(payload)
    return requests.get("http://localhost:%s/app/example_app/insert" % port, params={"payload": dumps(payload)})
