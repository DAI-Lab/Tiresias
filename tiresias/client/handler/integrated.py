
def handle_integrated(task, data):
    """
    The featurizer for a basic task is expected to produce a list of dictionaries such that each 
    dictionary contains the same set of keys.
    """
    assert type(data) == list, "Featurizers should return rows."
    keys = set(data[0].keys())
    for row in data:
        assert type(row) == dict, "Each row should be a dictionary."
        assert set(row.keys()) == keys, "Each dict should have the same keys."
    return data
