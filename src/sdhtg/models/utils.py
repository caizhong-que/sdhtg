from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def require_shape(tensor: Tensor, shape: tuple[int | None, ...], name: str) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(
            f"{name} must have {len(shape)} dimensions, got {tuple(tensor.shape)}"
        )
    for index, expected in enumerate(shape):
        if expected is not None and tensor.shape[index] != expected:
            raise ValueError(
                f"{name} dimension {index} must be {expected}, "
                f"got {tensor.shape[index]}"
            )


def validate_sequence_batch(batch: dict[str, Tensor]) -> tuple[int, int]:
    required = {
        "template_id",
        "entity_id",
        "action_id",
        "status_id",
        "delta_t",
        "action_change",
        "entity_change",
        "mask",
    }
    missing = required - batch.keys()
    if missing:
        raise KeyError(f"batch is missing tensors: {sorted(missing)}")

    mask = batch["mask"]
    require_shape(mask, (None, None), "mask")
    batch_size, steps = mask.shape

    for name in ("template_id", "entity_id", "action_id", "status_id"):
        require_shape(batch[name], (batch_size, steps), name)
        if batch[name].dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be an integer tensor")

    for name in ("delta_t", "action_change", "entity_change"):
        require_shape(batch[name], (batch_size, steps), name)

    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if not mask[:, 0].all():
        raise ValueError("every sequence must contain at least one valid event")
    if ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError("mask must be left-aligned without internal padding")
    if (batch["delta_t"][mask] < 0).any():
        raise ValueError("delta_t must be non-negative")
    return batch_size, steps


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


def masked_logsumexp_pool(
    values: Tensor,
    mask: Tensor,
    temperature: float,
    dim: int = 1,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    minimum = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, minimum)
    pooled = temperature * torch.logsumexp(masked / temperature, dim=dim)

    valid = mask.any(dim=dim)
    return torch.where(valid, pooled, torch.zeros_like(pooled))


def lengths_from_mask(mask: Tensor) -> Tensor:
    return mask.long().sum(dim=1)


def zero_padding(values: Tensor, mask: Tensor) -> Tensor:
    return values * mask.unsqueeze(-1).to(values.dtype)


def safe_normalize(weights: Tensor, dim: int, epsilon: float = 1e-8) -> Tensor:
    denominator = weights.sum(dim=dim, keepdim=True)
    return weights / denominator.clamp_min(epsilon)


def assert_finite(tensors: Iterable[Tensor], context: str) -> None:
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite tensor encountered in {context}")
