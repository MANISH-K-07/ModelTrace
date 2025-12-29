import torch
from .hooks import ActivationHook
from .metrics import calculate_sparsity

class ModelInspector:
    def __init__(self, model):
        self.model = model
        self.hook = ActivationHook()
        self.hook.register(model)

    def inspect(self, input_tensor):
        _ = self.model(input_tensor)

        # For now, sparsity is placeholder (step 2 will use activations)
        stats = {}
        for name, activation in self.hook.activations.items():
            stats[name] = float((activation == 0).float().mean())

        conv_s, fc_s, total_s = calculate_sparsity(stats)
        return conv_s, fc_s, total_s
