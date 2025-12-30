import torch
import numpy as np
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector


def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    # Run normal
    inspector.hook.activations.clear()
    with torch.no_grad():
        _ = model(x)
    acts_normal = {
        k: v.clone() for k, v in inspector.hook.activations.items()
    }

    # Run shifted
    inspector.hook.activations.clear()
    with torch.no_grad():
        _ = model(x_shifted)
    acts_shifted = inspector.hook.activations

    print("=== Layer Sensitivity Analysis ===")
    for layer in acts_normal:
        if layer in acts_shifted:
            delta = (acts_normal[layer] - acts_shifted[layer]).abs().mean().item()
            print(f"{layer}: sensitivity={delta:.4f}")


if __name__ == "__main__":
    main()
