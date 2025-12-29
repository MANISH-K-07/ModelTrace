import torch
from modeltrace.core.inspector import ModelInspector
from simple_cnn import SimpleCNN

def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    dummy_input = torch.randn(1, 1, 8, 8)
    stats = inspector.inspect_activations(dummy_input)

    print("=== Activation Statistics ===")
    for layer, s in stats.items():
        print(
            f"{layer}: mean={s['mean']:.4f}, std={s['std']:.4f}, max={s['max']:.4f}"
        )

if __name__ == "__main__":
    main()
