from tiresias.core import b64_encode, b64_decode
import tiresias.core.mechanisms as dp
import tiresias.core.machine_learning as ml
import tiresias.core.federated_learning as fl

def handle(query, data):
    dispatcher = {
        "basic": handle_basic,
        "bounded": handle_bounded,
        "machine_learning": handle_ml,
        "federated_learning": handle_fl
    }
    func = dispatcher.get(query["type"], lambda: ValueError("Unknown query type."))
    return func(query, data)

def handle_basic(query, data):
    dispatcher = {
        "mean": dp.mean,
        "median": dp.median,
        "count": dp.count,
        "sum": dp.sum
    }
    func = dispatcher.get(query["aggregator"], lambda: ValueError("Unknown aggregator."))
    return func(data, query["epsilon"])

def handle_bounded(query, data):
    return data

def handle_ml(query, data):
    return ml.compute(query["aggregator"], data)

def handle_fl(query, data):
    model = fl.aggregate(query["aggregator"], b64_decode(query["weights"]), [b64_decode(d) for d in data])
    return b64_encode(model)
