from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.GELU(), nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
        )
    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(self.network(value), dim=-1)


def supervised_info_nce(
    first: Tensor,
    second: Tensor,
    labels: Tensor | None = None,
    temperature: float = 0.07,
    hard_negative_k: int | None = 64,
) -> Tensor:
    if first.shape != second.shape:
        raise ValueError("contrastive views must have identical shapes")
    batch = first.shape[0]
    if batch < 2:
        return first.sum() * 0.0
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    similarity = first @ second.T / temperature
    positive = similarity.diag()
    eye = torch.eye(batch, dtype=torch.bool, device=first.device)
    negative_mask = ~eye
    if labels is not None:
        labels = labels.view(-1)
        # Same-label normal sessions may be workflow-aware positives; they must not
        # be treated as hard negatives. Anomalies remain instance positives only.
        same_normal = (labels[:, None] == 0) & (labels[None, :] == 0)
        negative_mask &= ~same_normal
    losses = []
    for index in range(batch):
        negatives = similarity[index][negative_mask[index]]
        if negatives.numel() == 0:
            continue
        if hard_negative_k and negatives.numel() > hard_negative_k:
            negatives = torch.topk(negatives, hard_negative_k).values
        denominator = torch.logsumexp(torch.cat((positive[index:index+1], negatives)), dim=0)
        losses.append(denominator-positive[index])
    return torch.stack(losses).mean() if losses else first.sum()*0.0
