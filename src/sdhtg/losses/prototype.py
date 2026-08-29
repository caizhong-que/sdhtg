from __future__ import annotations
import torch
from torch import Tensor


def prototype_margin_loss(distance: Tensor, target: Tensor, margin: float = 0.5) -> Tensor:
    target = target.to(distance.dtype)
    normal = (1.0 - target) * distance
    anomaly = target * torch.relu(margin - distance)
    return (normal + anomaly).mean()


def prototype_balance_loss(
    distances: Tensor,
    target: Tensor,
    temperature: float = 0.1,
    epsilon: float = 1e-8,
) -> Tensor:
    """Differentiable prototype balance loss (manuscript Eq. 63-65).

    q_{i,k} = softmax(-d_{i,k}/tau_p), averaged over normal samples only;
    L_bal = sum_k qbar_k * log(K * qbar_k + eps). Encourages prototypes to be
    used roughly equally without forcing per-sample uniform assignment.
    """
    target = target.to(distances.dtype)
    temperature = max(float(temperature), 1e-4)
    q = torch.softmax(-distances / temperature, dim=-1)
    normal = (1.0 - target).bool()
    if not normal.any():
        return distances.sum() * 0.0
    q_bar = q[normal].mean(dim=0)
    num_prototypes = q.shape[-1]
    return (q_bar * torch.log(num_prototypes * q_bar + epsilon)).sum()
