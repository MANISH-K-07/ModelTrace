from .hooks import ActivationHook

class ModelInspector:
    def __init__(self, model):
        self.model = model
        self.hook = ActivationHook()
        self.hook.register(model)

    def get_activations(self):
        return self.hook.activations
