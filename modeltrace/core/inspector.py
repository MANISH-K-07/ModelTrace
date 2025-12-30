import torch
import numpy as np
from .hooks import ActivationHook
from .metrics import calculate_sparsity


class ModelInspector:
    def __init__(self, model):
        self.model = model
        self.model.eval()

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
        acts_a = {k: v.clone() for k, v in self.hook.activations.items()}

        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(input_b)
        acts_b = self.hook.activations

        drift = []
        for k in acts_a:
            if k in acts_b:
                drift.append((acts_a[k] - acts_b[k]).abs().mean().item())

        return float(np.mean(drift)) if drift else 0.0

    # ==================================
    # Experiment 5: Cross-model Regression
    # ==================================
    def compute_model_regression(self, input_tensor, other_inspector):
        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(input_tensor)
        acts_self = self.hook.activations.copy()

        other_inspector.hook.activations.clear()
        with torch.no_grad():
            _ = other_inspector.model(input_tensor)
        acts_other = other_inspector.hook.activations

        drift = []
        for k in acts_self:
            if k in acts_other:
                drift.append((acts_self[k] - acts_other[k]).abs().mean().item())

        return float(np.mean(drift)) if drift else 0.0

    # ==================================
    # Experiment 6: Layer Sensitivity
    # ==================================
    def layer_sensitivity(self, input_a, input_b):
        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(input_a)
        acts_a = {k: v.clone() for k, v in self.hook.activations.items()}

        self.hook.activations.clear()
        with torch.no_grad():
            _ = self.model(input_b)
        acts_b = self.hook.activations

        sensitivity = {}
        for k in acts_a:
            if k in acts_b:
                sensitivity[k] = (acts_a[k] - acts_b[k]).abs().mean().item()

        return sensitivity

    # ==================================
    # Experiment 7: Failure Attribution
    # ==================================
    def failure_attribution(self, input_a, input_b):
        with torch.no_grad():
            out_a = self.model(input_a)
            out_b = self.model(input_b)

        output_shift = (out_a - out_b).abs().mean().item()
        sensitivities = self.layer_sensitivity(input_a, input_b)

        total = sum(sensitivities.values()) + 1e-8
        contributions = {
            k: v / total for k, v in sensitivities.items()
        }

        contributions = dict(
            sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        )

        return output_shift, contributions

    # ==================================
    # Experiment 8: Robustness Score
    # ==================================
    def robustness_score(self, input_a, input_b):
        normal_mag, shifted_mag = self.stress_test_distribution_shift(
            input_a, input_b
        )
        dist_shift = abs(shifted_mag - normal_mag)
        drift = self.compute_drift(input_a, input_b)
        sensitivities = self.layer_sensitivity(input_a, input_b)
        layer_sens = float(np.mean(list(sensitivities.values())))

        score = 0.4 * dist_shift + 0.4 * drift + 0.2 * layer_sens

        return {
            "distribution_shift": dist_shift,
            "activation_drift": drift,
            "layer_sensitivity": layer_sens,
            "final_score": score,
        }
