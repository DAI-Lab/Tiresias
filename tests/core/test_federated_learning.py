import torch
import tiresias.core.federated_learning as fl

def test_linear():
    spec = {
        "lr": 1e-4,
        "model": "Linear",
        "inputs": ["x0", "x1"],
        "outputs": ["y"],
        "loss": "MSE"
    }
    initial_weights = fl.random_weights(spec)
    old_model = fl.build_model(spec, initial_weights)

    gradients = []
    for _ in range(16):
        gradients.append(fl.gradients(spec, initial_weights, [{"x0": 0.0, "x1": 1.0, "y": 2.0}], epsilon=10.0))
    new_model = fl.aggregate(spec, initial_weights, gradients)
    
    old_loss = (old_model(torch.tensor([[0.0, 1.0]])) - 2.0) ** 2
    new_loss = (new_model(torch.tensor([[0.0, 1.0]])) - 2.0) ** 2
    assert new_loss.item() < old_loss.item()

def test_multilayer_perceptron():
    spec = {
        "lr": 1e-4,
        "model": "MultilayerPerceptron",
        "shape": [10, 10],
        "inputs": ["x0", "x1"],
        "outputs": ["y"],
        "loss": "MSE",
        "activation": "ReLU",
    }
    initial_weights = fl.random_weights(spec)
    old_model = fl.build_model(spec, initial_weights)

    gradients = []
    for _ in range(16):
        gradients.append(fl.gradients(spec, initial_weights, [{"x0": 0.0, "x1": 1.0, "y": 2.0}], epsilon=10.0))
    new_model = fl.aggregate(spec, initial_weights, gradients)
    
    old_loss = (old_model(torch.tensor([[0.0, 1.0]])) - 2.0) ** 2
    new_loss = (new_model(torch.tensor([[0.0, 1.0]])) - 2.0) ** 2
    assert new_loss.item() < old_loss.item()
