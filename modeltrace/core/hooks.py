import torch

class ActivationHook:
    def __init__(self):
        self.activations = {}

    def register(self, model):
        for name, layer in model.named_modules():
            layer.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, name):
        def hook(module, input, output):
            if torch.is_tensor(output):
                self.activations[name] = output.detach().cpu()
        return hook
