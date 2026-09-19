from __future__ import annotations

import re
from typing import Any

import torch


EXPLICIT_STATUS_RE = re.compile(
    r"\b(failed|failure|fatal|error|timeout|denied|invalid|unavailable|"
    r"exception|corrupt|abort)\w*\b",
    re.I,
)


def status_word_mask_from_vocab(vocab_status: dict[str, int]) -> list[int]:
    """Return status IDs whose token contains an explicit anomaly word."""
    reverse = {value: key for key, value in vocab_status.items()}
    return [
        value
        for value in range(len(vocab_status))
        if EXPLICIT_STATUS_RE.search(str(reverse.get(value, "")))
    ]


def shuffle_entity_ids(
    batch: dict[str, Any],
    entity_vocab_size: int,
    seed: int,
) -> None:
    """Permute entity IDs while preserving their frequency distribution."""
    if entity_vocab_size <= 2:
        return
    # Permute only the non-special ids and keep PAD (0) / UNK (1) fixed.
    # Assigning permutation[0] = 0 afterwards would break the bijection (two
    # ids would collide and two others would disappear), so the special ids
    # are excluded from the permutation instead.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    special = torch.arange(2)
    rest = torch.arange(2, entity_vocab_size)
    shuffled_rest = rest[torch.randperm(rest.numel(), generator=generator)]
    permutation = torch.cat((special, shuffled_rest)).to(
        batch["entity_id"].device
    )
    batch["entity_id"] = permutation[batch["entity_id"]]


def mask_entity_to_unk(batch: dict[str, Any]) -> None:
    batch["entity_id"] = batch["entity_id"].masked_fill(batch["mask"], 1)


def mask_explicit_status_words(
    batch: dict[str, Any],
    status_word_mask: list[int],
) -> None:
    if not status_word_mask:
        return
    mask_ids = torch.tensor(
        status_word_mask, dtype=torch.long, device=batch["status_id"].device
    )
    is_explicit = torch.isin(batch["status_id"], mask_ids) & batch["mask"]
    batch["status_id"] = batch["status_id"].masked_fill(is_explicit, 1)
