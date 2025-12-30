import torch
from modeltrace.core.inspector import ModelInspector
from simple_cnn import SimpleCNN

def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    dummy_inputs = torch.randn(10, 1, 8, 8)
    conv_vals = []
    fc_vals = []

    for x in dummy_inputs:
        conv_s, fc_s, _ = inspector.inspect(x.unsqueeze(0))
        conv_vals.append(conv_s)
        fc_vals.append(fc_s)

    print("=== Sparsity Analysis ===\n")
    print(f"Conv layer sparsity across batch: {sum(conv_vals)/len(conv_vals)}")
    print(f"FC layer sparsity across batch: {sum(fc_vals)/len(fc_vals)}")
    print("")

if __name__ == "__main__":
    main()
