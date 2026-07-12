from __future__ import annotations

from dataclasses import replace

import torch

from sdhtg.models.config import SDHTGModelConfig


def tiny_config(**changes) -> SDHTGModelConfig:
    base = SDHTGModelConfig(
        template_vocab_size=64,
        entity_vocab_size=32,
        action_vocab_size=32,
        status_vocab_size=64,
        hidden_dim=32,
        strategy_dim=16,
        time_dim=8,
        level_dim=8,
        num_gru_layers=1,
        graph_layers=2,
        graph_heads=4,
        dropout=0.0,
        num_normal_prototypes=4,
        local_temporal_radius={"status": 2, "action": 2, "entity": 1},
        semantic_neighbors={"status": 2, "action": 2, "entity": 1},
    )
    result = replace(base, **changes)
    result.validate()
    return result


def make_batch(
    lengths=(6, 4),
    total_steps: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    batch_size = len(lengths)
    steps = total_steps or max(lengths)
    if steps < max(lengths):
        raise ValueError("total_steps is shorter than a sequence")

    generator = torch.Generator(device="cpu").manual_seed(2026)
    mask = torch.zeros(batch_size, steps, dtype=torch.bool)
    for index, length in enumerate(lengths):
        mask[index, :length] = True

    def ids(high: int) -> torch.Tensor:
        values = torch.randint(2, high, (batch_size, steps), generator=generator)
        return values.masked_fill(~mask, 0)

    delta = torch.rand(batch_size, steps, generator=generator) * 20
    delta[:, 0] = 0
    delta = delta.masked_fill(~mask, 0)

    action = ids(16)
    entity = ids(12)
    action_change = torch.zeros(batch_size, steps)
    entity_change = torch.zeros(batch_size, steps)
    action_change[:, 0] = 1
    entity_change[:, 0] = 1
    action_change[:, 1:] = (action[:, 1:] != action[:, :-1]).float()
    entity_change[:, 1:] = (entity[:, 1:] != entity[:, :-1]).float()
    action_change = action_change.masked_fill(~mask, 0)
    entity_change = entity_change.masked_fill(~mask, 0)

    batch = {
        "template_id": ids(48),
        "entity_id": entity,
        "action_id": action,
        "status_id": ids(48),
        "delta_t": delta,
        "action_change": action_change,
        "entity_change": entity_change,
        "mask": mask,
    }
    return {key: value.to(device) for key, value in batch.items()}
