import torch
from modeltrace.core.inspector import ModelInspector
from simple_cnn import SimpleCNN

model = SimpleCNN()
inspector = ModelInspector(model)

dummy_input = torch.randn(1, 1, 28, 28)
_ = model(dummy_input)

activations = inspector.get_activations()

for layer, act in activations.items():
    print(layer, "sparsity:", activation_sparsity(act))
