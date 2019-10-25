import pickle
import codecs

def b64_encode(obj):
    """
    Take an arbitrary pickle-able object and encode it as a base64 string which
    can be transmitted as part of a JSON object.
    """
    return codecs.encode(pickle.dumps(obj), "base64").decode()

def b64_decode(obj):
    """
    Decode the given base64 string and attempt to unpickle it to recover the 
    original Python object.
    """
    return pickle.loads(codecs.decode(obj.encode(), "base64"))
