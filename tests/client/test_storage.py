from tiresias.client import storage

def test_initialize(tmpdir):
    storage.initialize(tmpdir)
    assert tmpdir.ensure("metadata.db"), "Metadata database was not created."

def test_register_app(tmpdir):
    storage.initialize(tmpdir)
    storage.register_app(tmpdir, "example_app", {
        "tableA": {
            "description": "This table contains A.",
            "columns": {
                "column1": {
                    "type": "float",
                    "description": "This column contains 1."
                },
                "column2": {
                    "type": "float",
                    "description": "This column contains 2."
                }
            }
        }
    })
    app_columns = storage.app_columns(tmpdir)
    assert len(app_columns) == 2
    for app_column in app_columns:
        assert app_column["app_name"] == 'example_app'
        assert app_column["table_name"] == 'tableA'
        assert app_column["column_name"] in ["column1", "column2"]

def test_insert_payload_1(tmpdir):
    storage.initialize(tmpdir)
    storage.register_app(tmpdir, "example_app", {
        "tableA": {
            "description": "This table contains A.",
            "columns": {
                "some_var": {
                    "type": "float",
                    "description": "This column contains 1."
                },
                "other_var": {
                    "type": "float",
                    "description": "This column contains 2."
                }
            }
        }
    })
    storage.insert_payload(tmpdir, "example_app", {
        "tableA": [
            {"some_var": 0.0, "other_var": 1.0},
            {"some_var": 10.0, "other_var": 1.0},
        ]
    })
    data = storage.execute_sql(tmpdir, "SELECT * FROM example_app.tableA")
    assert len(data) == 2, "Expected two rows of data"
    for row in data:
        assert "some_var" in row
        assert "other_var" in row
    assert sum(row["some_var"] for row in data) == 10.0
    assert sum(row["other_var"] for row in data) == 2.0

def test_insert_payload_2(tmpdir):
    storage.initialize(tmpdir)

    storage.register_app(tmpdir, "app1", {
        "tableA": {
            "description": "This table contains A.",
            "columns": {
                "some_var": {
                    "type": "float",
                    "description": "This column contains 1."
                }
            }
        }
    })
    storage.insert_payload(tmpdir, "app1", {
        "tableA": [
            {"some_var": 1.0},
            {"some_var": 2.0},
            {"some_var": 3.0},
            {"some_var": 4.0},
            {"some_var": 5.0},
        ]
    })

    storage.register_app(tmpdir, "app2", {
        "tableA": {
            "description": "This table contains A.",
            "columns": {
                "some_var": {
                    "type": "float",
                    "description": "This column contains 1."
                }
            }
        },
        "tableB": {
            "description": "This table contains B.",
            "columns": {
                "some_var": {
                    "type": "float",
                    "description": "This column contains 1."
                },
                "other_var": {
                    "type": "float",
                    "description": "This column contains 2."
                }
            }
        },
    })
    storage.insert_payload(tmpdir, "app2", {
        "tableA": [
            {"some_var": 3.0},
        ],
        "tableB": [
            {"some_var": 0.0, "other_var": 1.0},
            {"some_var": 10.0, "other_var": 1.0},
        ]
    })

    data = storage.execute_sql(tmpdir, "SELECT * FROM app1.tableA")
    assert len(data) == 5, "Expected five rows of data"

    data = storage.execute_sql(tmpdir, "SELECT * FROM app2.tableA")
    assert len(data) == 1, "Expected one row of data"

    data = storage.execute_sql(tmpdir, "SELECT * FROM app2.tableB")
    assert len(data) == 2, "Expected two rows of data"
