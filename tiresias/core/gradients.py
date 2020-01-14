import torch
import numpy as np

def get_gradients(model, epsilon, delta=0.01, max_norm=1.0):
    """
    This function extracts a differentially private copy of the gradients 
    from a PyTorch model where the loss has already been backpropagated through
    the neural network.
    """
    scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    # Compute the gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** (1. / 2)

    # Clip and add noise
    gradients = []
    for p in model.parameters():
        if p.requires_grad:
            grad = p.grad.data.clone()
            grad /= max(1.0, total_norm / max_norm)
            grad += torch.normal(
                torch.zeros(p.grad.data.size()), 
                torch.zeros(p.grad.data.size()) + scale * max_norm
            )
            gradients.append(grad)
    return gradients

def put_gradients(model, gradients):
    """
    This function takes the gradients extracted by the `get_gradients` 
    function and loads them back into the model.
    """
    for p, g in zip(model.parameters(), gradients):
        if p.requires_grad:
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
