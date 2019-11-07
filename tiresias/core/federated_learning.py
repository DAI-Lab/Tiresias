"""
The `tiresias.core.federated_learning` module provides functions to help 
implement differentially private federated learning in PyTorch. It provides
methods for estimating differentially private gradients and using 
aggregated estimates of the gradients to train a PyTorch model.
"""
# pylint: disable=no-member
# pylint: disable=not-callable
import torch
import numpy as np

def random_weights(spec):
    """
    This function builds the model and returns the random weights that it was
    initialized with. During the first iteration of a federated learning query,
    researchers are expected to use this function to generate the initial set
    of weights for the model.
    """
    return build_model(spec).state_dict()

def build_model(spec, weights=False):
    """
    This function constucts a PyTorch model from the given model specifications
    and optionally initializes it with the given weights.
    """
    if spec["model"] == "Linear":
        model = torch.nn.Linear(len(spec["inputs"]), len(spec["outputs"]))
    elif spec["model"] == "MultilayerPerceptron":
        shape = [len(spec["inputs"])] + spec["shape"] + [len(spec["outputs"])]
        layers = []
        for i, j in zip(shape, shape[1:]):
            layers.append(torch.nn.Linear(i, j))
            if spec["activation"] == "ReLU":
                layers.append(torch.nn.ReLU())
            else:
                raise ValueError(spec["activation"])
        layers.pop(-1)
        model = torch.nn.Sequential(*layers)
    else:
        raise ValueError(spec["model"])
    if weights:
        model.load_state_dict(weights)
    return model

def backpropagate(spec, model, data):
    """
    This function uses the model specification to feed the training data to the 
    model and backpropagates the gradients through the network. The gradients 
    are accumulated automatically by PyTorch.
    """
    x = torch.tensor([[d[var] for var in spec["inputs"]] for d in data])
    y = torch.tensor([[d[var] for var in spec["outputs"]] for d in data])
    if spec["loss"] == "MSE":
        loss = torch.mean((y - model(x))**2)
    else:
        raise ValueError()
    loss.backward()

def gradients(spec, weights, data, epsilon):
    """
    This function takes in the model specification, current weights, some 
    data points, and an epsilon value. It computes a differentially private
    estimate of the gradient using the method proposed in [1].

    [1] https://arxiv.org/pdf/1607.00133.pdf
    """
    model = build_model(spec, weights)
    backpropagate(spec, model, data)
    return get_gradients(model, epsilon)

def aggregate(spec, weights, list_of_gradients):
    """
    This function takes in the model specification, current weights, and a list 
    of gradient updates. It returns a new PyTorch model where the gradient 
    updates have been applied to the weights.
    """
    model = build_model(spec, weights)
    model.zero_grad()
    put_gradients(model, merge_gradients(list_of_gradients))
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    optimizer.step()
    return model

def get_gradients(model, epsilon, delta=0.01):
    """
    This function extracts a differentially private copy of the gradients 
    from a PyTorch model where the loss has already been backpropagated through
    the neural network.
    """
    # Clip the gradient
    max_norm = 1.0
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    for p in model.parameters():
        p.grad.data = p.grad.data / max(1.0, total_norm / max_norm)

    # Add noise to the gradient
    scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    for p in model.parameters():
        p.grad.data += torch.normal(torch.zeros(p.grad.data.size()), torch.zeros(p.grad.data.size()) + (scale**2) * (max_norm**2))
    
    # Return a list containing the gradients
    gradients = []
    for p in model.parameters():
        gradients.append(p.grad.data.clone())
    return gradients

def put_gradients(model, gradients):
    """
    This function takes the gradients extracted by the `get_gradients` 
    function and loads them back into the model.
    """
    for p, g in zip(model.parameters(), gradients):
        p.grad = g

def merge_gradients(list_of_gradients):
    """
    This function takes a list of gradients (e.g. a list of the objects 
    that are returned by the `get_gradients` function) and merges them into 
    a single gradient update which can be passed to `put_gradients`.
    """
    accumulator = list_of_gradients[0]
    for gradients in list_of_gradients[1:]:
        for i, g in enumerate(gradients):
            accumulator[i] += g
    for g in accumulator:
        g /= len(list_of_gradients)
    return accumulator
