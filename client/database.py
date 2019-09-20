import os
import sqlite3
import argparse
from json import loads, dumps
from bottle import route, run, request, response

def dict_factory(cursor, row):
    """
    This function extracts the column names from the SQLite cursor and returns
    a dictionary version of the row.
    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def initialize():
    """
    This function creates the data directory if necessary and initializes the 
    metadata database.
    """
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    connection = sqlite3.connect(os.path.join(args.data_dir, "metadata.db"))
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

@route('/')
def index():
    """
    This REST endpoint returns a JSON array containing a list of the columns stored on the device.
    """
    connection = sqlite3.connect(os.path.join(args.data_dir, "metadata.db"))
    connection.row_factory = dict_factory
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM app_columns")
    rows = cursor.fetchall()
    cursor.close()

    response.content_type = "application/json"
    return dumps(rows, indent=2)

@route('/query')
def query():
    """
    This REST endpoint accepts a `sql` parameter which contains the SQL query. It attaches all the
    application databases to the primary metadata database and executes the query. Note that this 
    endpoint does not perform any security checks.
    """
    connection = sqlite3.connect(os.path.join(args.data_dir, "metadata.db"))
    connection.row_factory = dict_factory
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM apps")
    for app in cursor.fetchall():
        cursor.execute("ATTACH DATABASE '%s' AS %s" % (os.path.join(args.data_dir, app["app_name"] + ".db"), app["app_name"]))

    cursor.execute(request.params.get("sql"))
    rows = cursor.fetchall()
    cursor.close()

    response.content_type = "application/json"
    return dumps(rows, indent=2)

@route('/app/<app_name>/register')
def register(app_name):
    """
    This REST endpoint allows a new application to register by providing their database schema. The
    `schema` parameter is a JSON object. For example, a schema which defines two tables containing 
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
    schema = loads(request.params.get("schema"))

    # Update the metadata database
    connection = sqlite3.connect(os.path.join(args.data_dir, "metadata.db"))
    cursor = connection.cursor()
    cursor.execute("INSERT INTO apps VALUES (?)", (app_name,))
    for table_name, obj in schema.items():
        cursor.execute("INSERT INTO app_tables VALUES (?, ?, ?)", (app_name, table_name, obj["description"]))
        for column_name, obj in obj["columns"].items():
            cursor.execute("INSERT INTO app_columns VALUES (?, ?, ?, ?, ?)", (app_name, table_name, column_name, obj["type"], obj["description"]))
    cursor.close()
    connection.commit()
    connection.close()

    # Create the app database
    connection = sqlite3.connect(os.path.join(args.data_dir, "%s.db" % app_name))
    cursor = connection.cursor()
    for table_name, obj in schema.items():
        cursor.execute("CREATE TABLE %s (%s)" % (table_name, ",".join(obj["columns"].keys())))
    cursor.close()
    connection.commit()
    connection.close()
    return ""

@route('/app/<app_name>/insert')
def insert(app_name):
    """
    This REST endpoint allows an application to append rows to their database by submitting a JSON 
    object in the `update` field. The entire JSON object is considered a single transaction; for 
    example, an update which inserts two rows into `tableX` and one row into `tableY` is shown 
    below:

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
    update = loads(request.params.get("update"))

    connection = sqlite3.connect(os.path.join(args.data_dir, "%s.db" % app_name))
    cursor = connection.cursor()

    for table_name, rows in update.items():
        for row in rows:
            keys, values = [], []
            for k, v in row.items():
                keys.append(k)
                values.append(v)
            
            columns = ', '.join(keys)
            placeholders = ', '.join('?' * len(keys))
            sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, columns, placeholders)
            cursor.execute(sql, values)
    
    cursor.close()
    connection.commit()
    connection.close()
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help="The port to listen on.")
    parser.add_argument('--data_dir', type=str, default="/tmp/tiresias", help="The target data directory.")
    args = parser.parse_args()

    initialize()
    run(host='localhost', port=args.port)
