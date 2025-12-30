"""
Unified runner for all ModelTrace experiments.

Run:
    python examples/run_all_experiments.py
"""

import torch
from simple_cnn import SimpleCNN
from modeltrace.core.inspector import ModelInspector


def experiment1_sparsity(inspector):
    print("\n[Experiment 1] Sparsity Analysis")

    x = torch.randn(10, 1, 8, 8)
    conv_vals, fc_vals = [], []

    for sample in x:
        conv_s, fc_s, _ = inspector.inspect(sample.unsqueeze(0))
        conv_vals.append(conv_s)
        fc_vals.append(fc_s)

    print(f"Conv sparsity (avg): {sum(conv_vals)/len(conv_vals):.4f}")
    print(f"FC sparsity (avg):   {sum(fc_vals)/len(fc_vals):.4f}")


def experiment2_activation_stats(inspector):
    print("\n[Experiment 2] Activation Statistics")

    x = torch.randn(1, 1, 8, 8)
    stats = inspector.inspect_activations(x)

    for layer, s in stats.items():
        print(
            f"{layer}: mean={s['mean']:.4f}, "
            f"std={s['std']:.4f}, "
            f"max={s['max']:.4f}"
        )


def experiment3_distribution_shift(inspector):
    print("\n[Experiment 3] Distribution Shift Test")
    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    normal, shifted = inspector.stress_test_distribution_shift(x, x_shifted)
    print(f"Normal mean activation magnitude:  {normal:.4f}")
    print(f"Shifted mean activation magnitude: {shifted:.4f}")


def experiment4_activation_drift(inspector):
    print("\n[Experiment 4] Activation Drift")
    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    drift = inspector.compute_drift(x, x_shifted)
    print(f"Drift score: {drift:.4f}")


def experiment5_model_regression():
    print("\n[Experiment 5] Model Regression Test")

    model_v1 = SimpleCNN()
    model_v2 = SimpleCNN()

    inspector_v1 = ModelInspector(model_v1)
    inspector_v2 = ModelInspector(model_v2)

    x = torch.randn(1, 1, 8, 8)

    drift = inspector_v1.compute_model_regression(
        x, other_inspector=inspector_v2
    )
    print(f"Regression drift score: {drift:.4f}")


def experiment6_layer_sensitivity(inspector):
    print("\n[Experiment 6] Layer Sensitivity Analysis")
    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    inspector.hook.activations.clear()
    _ = inspector.model(x)
    acts_a = {k: v.clone() for k, v in inspector.hook.activations.items()}

    inspector.hook.activations.clear()
    _ = inspector.model(x_shifted)
    acts_b = inspector.hook.activations

    for layer in acts_a:
        if layer in acts_b:
            delta = (acts_a[layer] - acts_b[layer]).abs().mean().item()
            print(f"{layer}: sensitivity={delta:.4f}")


def experiment7_failure_attribution(inspector):
    print("\n[Experiment 7] Failure Attribution")
    x = torch.randn(1, 1, 8, 8)
    x_perturbed = x + 0.3 * torch.randn_like(x)

    inspector.hook.activations.clear()
    out_a = inspector.model(x)
    acts_a = {k: v.clone() for k, v in inspector.hook.activations.items()}

    inspector.hook.activations.clear()
    out_b = inspector.model(x_perturbed)
    acts_b = inspector.hook.activations

    output_shift = (out_a - out_b).abs().mean().item()
    print(f"Output shift magnitude: {output_shift:.4f}")

    contribs = []
    for layer in acts_a:
        if layer in acts_b:
            delta = (acts_a[layer] - acts_b[layer]).abs().mean().item()
            contribs.append((layer, delta))

    total = sum(v for _, v in contribs) + 1e-8
    contribs = sorted(
        [(l, v / total) for l, v in contribs],
        key=lambda x: x[1],
        reverse=True,
    )

    print("Layer-wise contribution:")
    for layer, score in contribs:
        print(f"{layer}: {score:.4f}")


def experiment8_robustness_score(inspector):
    print("\n[Experiment 8] Model Robustness Score")
    x = torch.randn(1, 1, 8, 8)
    x_shifted = x + 0.5 * torch.randn_like(x)

    normal, shifted = inspector.stress_test_distribution_shift(x, x_shifted)
    drift = inspector.compute_drift(x, x_shifted)

    inspector.hook.activations.clear()
    _ = inspector.model(x)
    acts_a = {k: v.clone() for k, v in inspector.hook.activations.items()}

    inspector.hook.activations.clear()
    _ = inspector.model(x_shifted)
    acts_b = inspector.hook.activations

    sens = [
        (acts_a[k] - acts_b[k]).abs().mean().item()
        for k in acts_a if k in acts_b
    ]

    robustness = (
        0.4 * abs(shifted - normal)
        + 0.4 * drift
        + 0.2 * float(sum(sens) / len(sens))
    )

    print(f"FINAL ROBUSTNESS SCORE: {robustness:.4f}")


def main():
    print("\n============================================")
    print("     Running ALL ModelTrace Experiments     ")
    print("============================================")

    model = SimpleCNN()
    inspector = ModelInspector(model)

    experiment1_sparsity(inspector)
    print("--------------------------------------------")

    experiment2_activation_stats(inspector)
    print("--------------------------------------------")

    experiment3_distribution_shift(inspector)
    print("--------------------------------------------")

    experiment4_activation_drift(inspector)
    print("--------------------------------------------")

    experiment5_model_regression()
    print("--------------------------------------------")

    experiment6_layer_sensitivity(inspector)
    print("--------------------------------------------")

    experiment7_failure_attribution(inspector)
    print("--------------------------------------------")

    experiment8_robustness_score(inspector)
    print("--------------------------------------------")

    print("\n✅ All experiments completed successfully.")
    print("")


if __name__ == "__main__":
    main()
