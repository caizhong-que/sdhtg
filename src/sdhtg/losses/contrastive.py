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
    negative_strategy: str = "hard",
    semantic_keys: list[tuple[set[int], set[int]]] | None = None,
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

    if negative_strategy == "none":
        return -torch.log(torch.sigmoid(positive)).mean()

    if negative_strategy == "supervised":
        if labels is None:
            raise ValueError("supervised negative strategy requires labels")
        labels = labels.view(-1)
        same_label = labels[:, None] == labels[None, :]
        losses = []
        for index in range(batch):
            positives = similarity[index][same_label[index]]
            negatives = similarity[index][~same_label[index]]
            if positives.numel() == 0 or negatives.numel() == 0:
                continue
            denominator = torch.logsumexp(
                torch.cat((positives, negatives)), dim=0
            )
            losses.append((denominator - positives).mean())
        return torch.stack(losses).mean() if losses else first.sum() * 0.0

    if negative_strategy not in {"random", "hard", "semi_hard", "semantic"}:
        raise ValueError(
            f"negative_strategy must be random/hard/semi_hard/semantic/none/"
            f"supervised, got {negative_strategy!r}"
        )
    if negative_strategy == "semantic" and semantic_keys is None:
        raise ValueError("semantic negative strategy requires semantic_keys")

    losses = []
    for index in range(batch):
        negative_mask = ~eye[index]
        if labels is not None:
            same_normal = (labels[index] == 0) & (labels == 0)
            negative_mask &= ~same_normal
        if negative_strategy == "semantic" and semantic_keys is not None:
            anchor_templates, anchor_entities = semantic_keys[index]
            overlaps = []
            for other in range(batch):
                templates, entities = semantic_keys[other]
                overlap = bool(
                    (anchor_templates & templates) or (anchor_entities & entities)
                )
                overlaps.append(overlap)
            overlap_tensor = torch.tensor(
                overlaps, dtype=torch.bool, device=first.device
            )
            negative_mask &= ~overlap_tensor
        negatives = similarity[index][negative_mask]
        if negatives.numel() == 0:
            continue
        k = int(hard_negative_k or negatives.numel())
        k = min(k, negatives.numel())
        if negative_strategy == "random":
            selected = negatives[torch.randperm(negatives.numel(), device=first.device)[:k]]
        elif negative_strategy == "semi_hard":
            below = negatives[negatives < positive[index]]
            if below.numel() == 0:
                selected = torch.topk(negatives, k).values
            else:
                selected = torch.topk(below, min(k, below.numel())).values
        else:
            selected = torch.topk(negatives, k).values
        denominator = torch.logsumexp(
            torch.cat((positive[index : index + 1], selected)), dim=0
        )
        losses.append(denominator-positive[index])
    return torch.stack(losses).mean() if losses else first.sum()*0.0
