import sys
import os

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from examples.simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector


def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    dummy_inputs = torch.randn(10, 1, 8, 8)

    conv_s_list = []
    fc_s_list = []

    for inp in dummy_inputs:
        conv_s, fc_s, total_s = inspector.inspect(inp.unsqueeze(0))
        conv_s_list.append(conv_s)
        fc_s_list.append(fc_s)

    print("=== Sparsity Analysis ===")
    print(f"Conv layer sparsity across batch: {sum(conv_s_list)/len(conv_s_list)}")
    print(f"FC layer sparsity across batch: {sum(fc_s_list)/len(fc_s_list)}")


if __name__ == "__main__":
    main()
