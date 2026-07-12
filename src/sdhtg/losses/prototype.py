from __future__ import annotations
import torch
from torch import Tensor


def prototype_margin_loss(distance: Tensor, target: Tensor, margin: float = 0.5) -> Tensor:
    target = target.to(distance.dtype)
    normal = (1.0 - target) * distance
    anomaly = target * torch.relu(margin - distance)
    return (normal + anomaly).mean()
