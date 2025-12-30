import torch
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector


def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    sensitivity = inspector.layer_sensitivity(x, x_shifted)

    print("=== Layer Sensitivity Analysis ===\n")
    for layer, delta in sensitivity.items():
        print(f"{layer}: sensitivity={delta:.4f}")
    print("")


if __name__ == "__main__":
    main()
