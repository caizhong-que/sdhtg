from __future__ import annotations
import torch
from torch import Tensor


def boundary_regularization(
    action: Tensor,
    entity: Tensor,
    mask: Tensor,
    action_rate: float,
    entity_rate: float,
) -> dict[str, Tensor]:
    valid = mask.to(action.dtype)
    denominator = valid.sum().clamp_min(1.0)
    entropy = -(
        action.clamp(1e-6, 1-1e-6) * action.clamp(1e-6, 1-1e-6).log()
        + (1-action).clamp(1e-6, 1-1e-6) * (1-action).clamp(1e-6, 1-1e-6).log()
        + entity.clamp(1e-6, 1-1e-6) * entity.clamp(1e-6, 1-1e-6).log()
        + (1-entity).clamp(1e-6, 1-1e-6) * (1-entity).clamp(1e-6, 1-1e-6).log()
    )
    entropy = (entropy * valid).sum() / denominator
    rate = ((action * valid).sum()/denominator-action_rate).square()
    rate = rate + ((entity * valid).sum()/denominator-entity_rate).square()
    separation = (torch.relu(entity-action) * valid).sum()/denominator
    hierarchy = ((entity * (1-action)) * valid).sum()/denominator
    return {"entropy": entropy, "rate": rate, "separation": separation, "hierarchy": hierarchy}
