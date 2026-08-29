from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def effective_number_weights(class_counts: Tensor, beta: float = 0.9999) -> Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.numel() != 2 or (counts <= 0).any():
        raise ValueError("class_counts must contain two positive counts")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0,1)")
    weights = (1.0 - beta) / (1.0 - beta ** counts).clamp_min(1e-12)
    return weights / weights.sum() * 2.0


def class_balanced_focal_loss(
    logits: Tensor,
    targets: Tensor,
    class_counts: Tensor,
    beta: float = 0.9999,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    targets = targets.to(logits.dtype)
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if not 0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0,0.5)")
    smooth = targets * (1 - 2 * label_smoothing) + label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(targets.bool(), probability, 1.0 - probability)
    weights = effective_number_weights(class_counts, beta).to(logits.device)
    alpha = weights[targets.long()]
    return (alpha * (1.0 - pt).pow(gamma) * bce).mean()


def binary_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Plain binary cross-entropy without class weighting."""
    targets = targets.to(logits.dtype)
    smooth = targets * (1 - 2 * label_smoothing) + label_smoothing
    return F.binary_cross_entropy_with_logits(logits, smooth)


def weighted_bce_loss(
    logits: Tensor,
    targets: Tensor,
    class_counts: Tensor,
    beta: float = 0.9999,
    label_smoothing: float = 0.0,
) -> Tensor:
    """BCE weighted by effective-number class weights."""
    targets = targets.to(logits.dtype)
    smooth = targets * (1 - 2 * label_smoothing) + label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction="none")
    weights = effective_number_weights(class_counts, beta).to(logits.device)
    alpha = weights[targets.long()]
    return (alpha * bce).mean()


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Focal loss without class weights (alpha = 1)."""
    targets = targets.to(logits.dtype)
    smooth = targets * (1 - 2 * label_smoothing) + label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(targets.bool(), probability, 1.0 - probability)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def classification_loss(
    logits: Tensor,
    targets: Tensor,
    class_counts: Tensor,
    loss_type: str = "cb_focal",
    beta: float = 0.9999,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Dispatch among bce / weighted_bce / focal / cb_focal."""
    if loss_type == "bce":
        return binary_cross_entropy_loss(logits, targets, label_smoothing)
    if loss_type == "weighted_bce":
        return weighted_bce_loss(
            logits, targets, class_counts, beta, label_smoothing
        )
    if loss_type == "focal":
        return focal_loss(logits, targets, gamma, label_smoothing)
    if loss_type == "cb_focal":
        return class_balanced_focal_loss(
            logits, targets, class_counts, beta, gamma, label_smoothing
        )
    raise ValueError(
        f"loss_type must be bce/weighted_bce/focal/cb_focal, got {loss_type!r}"
    )
