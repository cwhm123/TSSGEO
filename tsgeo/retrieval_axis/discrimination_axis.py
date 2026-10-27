import numpy as np
import torch


def compute_discrimination_axis(
    a_activations: torch.Tensor,
    b_activations: torch.Tensor,
) -> torch.Tensor:
    if a_activations.ndim != 3 or b_activations.ndim != 3:
        raise ValueError("Expected (n, n_layers, hidden) for both inputs.")
    return a_activations.mean(dim=0) - b_activations.mean(dim=0)


def project_onto_axis(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> float:
    if axis.ndim != 2:
        raise ValueError("axis must be (n_layers, hidden).")

    if activations.ndim == 2:
        act = activations[layer].float()
    elif activations.ndim == 1:
        act = activations.float()
    else:
        raise ValueError("activations must be (n_layers, hidden) or (hidden,).")

    ax = axis[layer].float()
    if normalize:
        ax = ax / (ax.norm() + 1e-8)
    return float(act @ ax)


def project_batch_onto_axis(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> torch.Tensor:
    if activations.ndim != 3:
        raise ValueError("activations must be (batch, n_layers, hidden).")
    acts = activations[:, layer, :].float()
    ax = axis[layer].float()
    if normalize:
        ax = ax / (ax.norm() + 1e-8)
    return acts @ ax


def cosine_similarity_per_layer(v1: torch.Tensor, v2: torch.Tensor) -> np.ndarray:
    if v1.ndim != 2 or v2.ndim != 2:
        raise ValueError("v1 and v2 must be (n_layers, hidden).")
    v1 = v1.float()
    v2 = v2.float()
    v1n = v1 / (v1.norm(dim=1, keepdim=True) + 1e-8)
    v2n = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)
    return (v1n * v2n).sum(dim=1).detach().cpu().numpy()
