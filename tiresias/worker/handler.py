from tiresias.core import b64_encode, b64_decode
import tiresias.core.mechanisms as dp
import tiresias.core.machine_learning as ml
import tiresias.core.federated_learning as fl

def handle(query, data):
    """
    This function takes in a generic query and data and dispatches it to the 
    appropriate query handler.
    """
    dispatcher = {
        "basic": handle_basic,
        "bounded": handle_bounded,
        "machine_learning": handle_ml,
        "federated_learning": handle_fl
    }
    func = dispatcher.get(query["type"], lambda: ValueError("Unknown query type."))
    return func(query, data)

def handle_basic(query, data):
    """
    This function handles basic queries. The data is expected to be a list of 
    scalars, one from each user who contributed to the query, and this function
    applies the appropriate differentially private aggregation mechanism.
    
    See `tiresias.client.handler.handle_basic` for corresponding function on 
    the client side.
    """
    dispatcher = {
        "mean": dp.mean,
        "median": dp.median,
        "count": dp.count,
        "sum": dp.sum
    }
    func = dispatcher.get(query["aggregator"], lambda: ValueError("Unknown aggregator."))
    return func(data, query["epsilon"])

def handle_bounded(query, data):
    """
    This function handles bounded queries. No work needs to be done on the 
    server side for bounded queries as the differential privacy mechanisms are
    applied on the client side for this type of query.

    See `tiresias.client.handler.handle_bounded` for corresponding function on 
    the client side.
    """
    return data

def handle_ml(query, data):
    """
    This function handles machine learning queries. The machine learning 
    model implementations are provided `tiresias.core.machine_learning`.

    See `tiresias.client.handler.handle_ml` for corresponding function on 
    the client side.
    """
    return ml.compute(query["aggregator"], data)

def handle_fl(query, data):
    """
    This function handles federated learning queries. The federated learning 
    model implementations are provided `tiresias.core.federated_learning`. On
    the server side, this simply entails aggregating the gradients computed 
    by the clients.

    See `tiresias.client.handler.handle_fl` for corresponding function on 
    the client side.
    """
    model = fl.aggregate(query["aggregator"], b64_decode(query["weights"]), [b64_decode(d) for d in data])
    return b64_encode(model)
