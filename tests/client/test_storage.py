import tiresias.client.storage as storage

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
