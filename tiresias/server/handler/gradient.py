import torch
import numpy as np
from tiresias.core import b64_decode, b64_encode
from tiresias.core.gradients import merge_gradients, put_gradients

def handle_gradient(task, data):
    model = b64_decode(task["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=task["lr"])
    put_gradients(model, merge_gradients([b64_decode(g) for g in data]))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return b64_encode(model)
