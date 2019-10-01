import os
import sqlite3
import requests
from bottle import Bottle, request, response
from json import loads, dumps

def dict_factory(cursor, row):
    """
    This function extracts the column names from the SQLite cursor and returns
    a dictionary version of the row.
    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def initialize(data_dir):
    """
    This function creates the data directory if necessary and initializes the 
    metadata database.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    connection = sqlite3.connect(os.path.join(data_dir, "metadata.db"))
    cursor = connection.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS apps (app_name);
        CREATE TABLE IF NOT EXISTS app_tables (
            app_name, 
            table_name, 
            table_description
        );
        CREATE TABLE IF NOT EXISTS app_columns (
            app_name, 
            table_name, 
            column_name, 
            column_type, 
            column_description
        );
    """)
    cursor.close()
    connection.close()

def app_columns(data_dir):
    """
    Return a list of dictionaries describing the columns that are being collected.
    """
    connection = sqlite3.connect(os.path.join(data_dir, "metadata.db"))
    connection.row_factory = dict_factory

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM app_columns")
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    return rows

def execute_sql(data_dir, sql):
    """
    Execute the given query on the database, using "<app_name>.<table_name>" to specify tables.
    """
    connection = sqlite3.connect(os.path.join(data_dir, "metadata.db"))
    connection.row_factory = dict_factory
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT * FROM apps")
        for app in cursor.fetchall():
            cursor.execute("ATTACH DATABASE '%s' AS %s" % (os.path.join(data_dir, app["app_name"] + ".db"), app["app_name"]))

        cursor.execute(sql)
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    return rows

def validate_schema(schema):
    assert type(schema) == dict, "Expected schema to be a dictionary."
    for table_name, table in schema.items():
        assert type(table_name) == str, "Expected schema to be a str -> dict mapping"
        assert type(table) == dict, "Expected schema to be a str -> dict mapping"
        assert "description" in table, "Expected table to contain a description"
        assert "columns" in table, "Expected table to contain columns"
        for column_name, column in table["columns"].items():
            assert type(column_name) == str, "Expected each column to have a string name"
            assert "type" in column, "Expected column to contain a type"
            assert "description" in column, "Expected column to contain a description"
            assert column["type"] in ["int", "string", "float"], "Expected type to be in {int, string, float}"
            assert type(column["description"]) == str, "Expected description to be a string"

def register_app(data_dir, app_name, schema):
    """
    Register an application, which is defined by an app_name and a schema, and create the 
    appropriate tables in the database. For example, a schema which defines two tables containing 
    one and two columns, respectively, is shown here:

    schema = {
        "tableX": {
            "description": "This table contains X.",
            "columns": {
                "column1": {
                    "type": "float",
                    "description: "This column contains 1."
                }
            }
        },
        "tableY": {
            "description": "This table contains Y.",
            "columns": {
                "column1": {
                    "type": "float",
                    "description: "This column contains 1."
                },
                "column2": {
                    "type": "string",
                    "description: "This column contains 2."
                }
            }
        }
    }
    """
    # Validate the schema
    validate_schema(schema)

    # Update the metadata database
    connection = sqlite3.connect(os.path.join(data_dir, "metadata.db"))
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM apps WHERE app_name = ?", (app_name,))
        if cursor.fetchone():
            return ValueError("App already exists: %s" % app_name)
        cursor.execute("INSERT INTO apps VALUES (?)", (app_name,))
        for table_name, table in schema.items():
            cursor.execute("INSERT INTO app_tables VALUES (?, ?, ?)", (app_name, table_name, table["description"]))
            for column_name, column in table["columns"].items():
                cursor.execute("INSERT INTO app_columns VALUES (?, ?, ?, ?, ?)", (app_name, table_name, column_name, column["type"], column["description"]))
        cursor.close()
        connection.commit()
    finally:
        connection.close()

    # Create the app database
    try:
        connection = sqlite3.connect(os.path.join(data_dir, "%s.db" % app_name))
        cursor = connection.cursor()
        for table_name, table in schema.items():
            cursor.execute("CREATE TABLE %s (%s)" % (table_name, ",".join(table["columns"].keys())))
        cursor.close()
        connection.commit()
    finally:
        connection.close()

def validate_payload(payload):
    assert type(payload) == dict, "Expected payload to be a dictionary."
    for table_name, rows in payload.items():
        assert type(table_name) == str, "Expected payload to be a str -> [] mapping"
        assert type(rows) == list, "Expected payload to be a str -> [] mapping"
        for row in rows:
            assert type(row) == dict, "Expected each row in the update to be a dictionary"

def insert_payload(data_dir, app_name, payload):
    """
    The entire payload is considered a single transaction; for example, an update which inserts two 
    rows into `tableX` and one row into `tableY` is shown below:

    update = {
        "tableX": [
            {"column1": 0.0},
            {"column1": 0.0},
        ],
        "tableY": [{
            {"column1": 0.0, "column2": "hello"}
        }]
    }
    """
    connection = sqlite3.connect(os.path.join(data_dir, "%s.db" % app_name))
    cursor = connection.cursor()

    try:
        for table_name, rows in payload.items():
            for row in rows:
                keys, values = [], []
                for k, v in row.items():
                    keys.append(k)
                    values.append(v)
                
                columns = ', '.join(keys)
                placeholders = ', '.join('?' * len(keys))
                sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, columns, placeholders)
                cursor.execute(sql, values)
        connection.commit()
    finally:
        connection.close()

def run(data_dir="/tmp/tiresias", port=8000):
    api = Bottle()
    initialize(data_dir)
    api.config['data_dir'] = data_dir

    @api.route("/")
    def index():
        """
        This REST endpoint returns a JSON array containing a list of the columns stored on the device.
        """
        rows = app_columns(api.config['data_dir'])
        response.content_type = "application/json"
        return dumps(rows, indent=2)

    @api.route("/query")
    def query():
        """
        This REST endpoint accepts a `sql` parameter which contains the SQL query. It attaches all the
        application databases to the primary metadata database and executes the query. Note that this 
        endpoint does not perform any security checks.
        """
        rows = execute_sql(api.config['data_dir'], request.params.get("sql"))
        response.content_type = "application/json"
        return dumps(rows, indent=2)

    @api.route("/app/<app_name>/register")
    def register(app_name):
        """
        This REST endpoint allows a new application to register by providing their database schema. The
        `schema` parameter is a JSON object.
        """
        schema = loads(request.params.get("schema"))
        register_app(api.config['data_dir'], app_name, schema)
        return ""

    @api.route("/app/<app_name>/insert")
    def insert(app_name):
        """
        This REST endpoint allows an application to append rows to their database by submitting a JSON 
        object in the `payload` field.
        """
        payload = loads(request.params.get("payload"))
        insert_payload(api.config['data_dir'], app_name, payload)
        return ""

    api.run(host="localhost", port=port)
