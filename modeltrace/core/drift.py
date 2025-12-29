import torch

def activation_drift_score(activations_a, activations_b):
    """
    Computes mean L2 drift between two activation dictionaries
    """
    drift_scores = []

    for layer in activations_a:
        if layer in activations_b:
            a = activations_a[layer].float()
            b = activations_b[layer].float()
            drift = torch.norm(a - b) / a.numel()
            drift_scores.append(drift.item())

    return sum(drift_scores) / len(drift_scores) if drift_scores else 0.0
