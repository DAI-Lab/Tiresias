import torch
import numpy as np
from tiresias.core import b64_decode, b64_encode
from tiresias.core.gradients import get_gradients

def handle_gradient(task, data):
    """
    The featurizer for a basic task is expected to produce a list of dictionaries such that each 
    dictionary contains the same set of keys. This function process it into a X and Y matrix, 
    decodes the model and loss function, computes the gradients for the loss, and returns an 
    encoded copy of the gradients.
    """
    x = np.array([[row[var] for var in task["inputs"]] for row in data])
    y = np.array([[row[var] for var in task["output"]] for row in data])

    model = b64_decode(task["model"])
    loss = b64_decode(task["loss"])
    loss(torch.FloatTensor(y), model(torch.FloatTensor(x))).backward()
    return b64_encode(get_gradients(model, task["epsilon"], task["delta"]))
