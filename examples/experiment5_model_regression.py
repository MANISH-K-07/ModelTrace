import torch
import torch.nn as nn
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector

def main():
    model_v1 = SimpleCNN()
    model_v2 = SimpleCNN()

    # Simulate regression by perturbing model_v2
    with torch.no_grad():
        for p in model_v2.parameters():
            p.add_(0.05 * torch.randn_like(p))

    inspector_v1 = ModelInspector(model_v1)
    inspector_v2 = ModelInspector(model_v2)

    x = torch.randn(1, 1, 8, 8)

    drift = inspector_v1.compute_model_regression(
        x,
        other_inspector=inspector_v2
    )

    print("=== Model Regression Test ===")
    print(f"Regression drift score: {drift:.4f}")

if __name__ == "__main__":
    main()
