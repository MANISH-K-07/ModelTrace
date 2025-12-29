import torch
import numpy as np
from .hooks import ActivationHook
from .metrics import calculate_sparsity


class ModelInspector:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        # ONE hook system, reused everywhere
        self.hook = ActivationHook()
        self.hook.register(self.model)

    # ===============================
    # Experiment 1: Sparsity Analysis
    # ===============================
    def inspect(self, input_tensor):
        with torch.no_grad():
            _ = self.model(input_tensor)

        stats = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "weight"):
                weight = module.weight.detach().cpu()
                stats[name] = (weight == 0).float().mean().item()

        conv_s, fc_s, total_s = calculate_sparsity(stats)
        return conv_s, fc_s, total_s

    # =================================
    # Experiment 2: Activation Statistics
    # =================================
    def inspect_activations(self, input_tensor):
        self.hook.activations.clear()

        with torch.no_grad():
            _ = self.model(input_tensor)

        stats = {}
        for name, act in self.hook.activations.items():
            stats[name] = {
                "mean": act.mean().item(),
                "std": act.std().item(),
                "max": act.max().item(),
            }
        return stats

    # =====================================
    # Experiment 3: Distribution Shift Test
    # =====================================
    def stress_test_distribution_shift(self, normal_input, shifted_input):
        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(normal_input)

        normal_mag = np.mean(
            [v.abs().mean().item() for v in self.hook.activations.values()]
        )

        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(shifted_input)

        shifted_mag = np.mean(
            [v.abs().mean().item() for v in self.hook.activations.values()]
        )

        return normal_mag, shifted_mag

    # ==============================
    # Experiment 4: Activation Drift
    # ==============================
    def compute_drift(self, input_a, input_b):
        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(input_a)

        acts_a = {
            k: v.clone() for k, v in self.hook.activations.items()
        }

        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(input_b)

        acts_b = self.hook.activations

        drift = []
        for k in acts_a:
            if k in acts_b:
                drift.append((acts_a[k] - acts_b[k]).abs().mean().item())

        return float(np.mean(drift)) if drift else 0.0
