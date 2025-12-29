import torch

def activation_sparsity(tensor):
    if not torch.is_tensor(tensor):
        return None
    return (tensor == 0).float().mean().item()
