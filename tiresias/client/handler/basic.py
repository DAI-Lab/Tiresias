
def handle_basic(task, data):
    """
    The featurizer for a basic task is expected to produce a list of dictionaries such that each 
    dictionary contains a single key-value pair; this function flattens it into a list of values.
    """
    assert type(data) == list, "Featurizers should return rows."
    values = []
    for row in data:
        assert len(row) == 1, "Each row should only contain one value."
        values.append(list(row.values())[0])
    return values
