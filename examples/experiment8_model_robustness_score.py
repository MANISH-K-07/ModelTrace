import torch
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector

def main():
    model = SimpleCNN()
    inspector = ModelInspector(model)

    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    scores = inspector.robustness_score(x, x_shifted)

    print("=== Model Robustness Score ===\n")
    print(f"Distribution shift component: {scores['distribution_shift']:.4f}")
    print(f"Activation drift component:   {scores['activation_drift']:.4f}")
    print(f"Layer sensitivity component: {scores['layer_sensitivity']:.4f}")
    print(f"\nFINAL ROBUSTNESS SCORE: {scores['final_score']:.4f}")
    print("")

if __name__ == "__main__":
    main()
