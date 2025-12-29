import torch
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector
from modeltrace.core.metrics import activation_sparsity

model = SimpleCNN()
inspector = ModelInspector(model)

dummy_input = torch.randn(1, 1, 28, 28)
_ = model(dummy_input)

activations = inspector.get_activations()

for layer, act in activations.items():
    print(layer, "sparsity:", activation_sparsity(act))
