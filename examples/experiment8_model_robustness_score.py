import torch
import numpy as np
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector


def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    # ---- Metric 1: Distribution Shift ----
    normal_mag, shifted_mag = inspector.stress_test_distribution_shift(
        x, x_shifted
    )
    dist_shift_score = abs(shifted_mag - normal_mag)

    # ---- Metric 2: Activation Drift ----
    drift_score = inspector.compute_drift(x, x_shifted)

    # ---- Metric 3: Layer Sensitivity ----
    inspector.hook.activations.clear()
    with torch.no_grad():
        _ = model(x)
    acts_a = {
        k: v.clone() for k, v in inspector.hook.activations.items()
    }

    inspector.hook.activations.clear()
    with torch.no_grad():
        _ = model(x_shifted)
    acts_b = inspector.hook.activations

    layer_sensitivities = []
    for k in acts_a:
        if k in acts_b:
            layer_sensitivities.append(
                (acts_a[k] - acts_b[k]).abs().mean().item()
            )

    layer_sensitivity_score = float(np.mean(layer_sensitivities))

    # ---- Final Robustness Score ----
    robustness_score = (
        0.4 * dist_shift_score
        + 0.4 * drift_score
        + 0.2 * layer_sensitivity_score
    )

    print("=== Model Robustness Score ===")
    print(f"Distribution shift component: {dist_shift_score:.4f}")
    print(f"Activation drift component:   {drift_score:.4f}")
    print(f"Layer sensitivity component: {layer_sensitivity_score:.4f}")
    print(f"\nFINAL ROBUSTNESS SCORE: {robustness_score:.4f}")


if __name__ == "__main__":
    main()
