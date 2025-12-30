import torch
import numpy as np
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector


def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    # Base input
    x = torch.randn(1, 1, 8, 8)
    x_perturbed = x + 0.3 * torch.randn_like(x)

    # Run normal input
    inspector.hook.activations.clear()
    with torch.no_grad():
        out_normal = model(x)
    acts_normal = {
        k: v.clone() for k, v in inspector.hook.activations.items()
    }

    # Run perturbed input
    inspector.hook.activations.clear()
    with torch.no_grad():
        out_perturbed = model(x_perturbed)
    acts_perturbed = inspector.hook.activations

    # Output shift magnitude
    output_shift = (out_normal - out_perturbed).abs().mean().item()

    print("=== Failure Attribution Analysis ===")
    print(f"Output shift magnitude: {output_shift:.4f}")
    print("\nLayer-wise contribution:")

    contributions = []

    for layer in acts_normal:
        if layer in acts_perturbed:
            delta = (acts_normal[layer] - acts_perturbed[layer]).abs().mean().item()
            contributions.append((layer, delta))

    # Normalize contributions
    total = sum(v for _, v in contributions) + 1e-8
    contributions = [(l, v / total) for l, v in contributions]

    # Sort by importance
    contributions.sort(key=lambda x: x[1], reverse=True)

    for layer, score in contributions:
        print(f"{layer}: contribution={score:.4f}")


if __name__ == "__main__":
    main()
