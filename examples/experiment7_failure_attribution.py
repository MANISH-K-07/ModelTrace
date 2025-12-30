import torch
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector

def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    x = torch.randn(1, 1, 8, 8)
    x_perturbed = x + 0.3 * torch.randn_like(x)

    output_shift, contributions = inspector.failure_attribution(
        x, x_perturbed
    )

    print("=== Failure Attribution Analysis ===\n")
    print(f"Output shift magnitude: {output_shift:.4f}\n")

    for layer, score in contributions.items():
        print(f"{layer}: contribution={score:.4f}")
    print("")

if __name__ == "__main__":
    main()
