from tiresias.core.mechanisms import finite_categorical, bounded_continuous

def _ldp(value, bounds, epsilon):
    if bounds["type"] == "range":
        value = min(max(bounds["low"], value), bounds["high"])
        return bounded_continuous(value, bounds["low"], bounds["high"], epsilon)

    elif bounds["type"] == "set":
        if value not in bounds["values"]:
            value = bounds["default"]
        domain = set(bounds["values"])
        domain.add(bounds["default"])
        return finite_categorical(value, domain, epsilon)

    return ValueError("Unknown bounds.")

def handle_bounded(task, data):
    """
    The featurizer for a basic task is expected to produce a list of dictionaries such that the 
    bounds for each value is given in the task. This applies local differential privacy to each
    value independently (e.g. so each value on its own is differentially private).
    """
    dp_rows = []
    for row in data:
        dp_row = {}
        for key, value in row.items():
            bounds = task["bounds"][key]
            dp_row[key] = _ldp(value, bounds, task["epsilon"])
        dp_rows.append(dp_row)
    return dp_rows
