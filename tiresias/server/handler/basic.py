import tiresias.core.mechanisms as mechanisms

def handle_basic(task, data):
    dispatcher = {
        "mean": mechanisms.mean,
        "median": mechanisms.median,
        "count": mechanisms.count,
        "sum": mechanisms.sum
    }
    if task["aggregator"] not in dispatcher:
        raise ValueError("Unknown aggregator.")
    func = dispatcher[task["aggregator"]]

    values = []
    for row in data:
        values.extend(row)
    return func(values, task["epsilon"], task["delta"])
