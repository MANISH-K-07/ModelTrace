import torch
from modeltrace.core.inspector import ModelInspector
from simple_cnn import SimpleCNN

def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    normal_input = torch.randn(1, 1, 8, 8)
    shifted_input = torch.randn(1, 1, 8, 8) + 3.0  # distribution shift

    normal_mag, shifted_mag = inspector.stress_test_distribution_shift(
        normal_input, shifted_input
    )

    print("=== Distribution Shift Stress Test ===")
    print(f"Normal mean activation magnitude: {normal_mag:.4f}")
    print(f"Shifted mean activation magnitude: {shifted_mag:.4f}")

if __name__ == "__main__":
    main()
