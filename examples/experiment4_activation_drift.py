import torch
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector

def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    normal_input = torch.randn(1, 1, 8, 8)
    shifted_input = torch.randn(1, 1, 8, 8) * 3.0

    drift = inspector.compute_drift(normal_input, shifted_input)

    print("=== Activation Drift Score ===")
    print(f"Drift score: {drift:.4f}")

if __name__ == "__main__":
    main()
