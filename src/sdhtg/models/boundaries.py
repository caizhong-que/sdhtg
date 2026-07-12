from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import SDHTGModelConfig
from .utils import zero_padding


@dataclass
class BoundaryOutput:
    action_probability: Tensor
    entity_probability: Tensor
    action_logit: Tensor
    entity_conditional_logit: Tensor


class BoundaryHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class NestedSoftBoundaryNetwork(nn.Module):
    """
    Predicts boundary-before-event probabilities.

    p_entity = p_action * sigmoid(entity_conditional_logit), enforcing
    0 <= p_entity <= p_action <= 1 exactly, rather than by a penalty.
    """

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        strategy = config.strategy_dim
        input_dim = hidden * 3 + strategy + 3

        self.config = config
        self.action_head = BoundaryHead(input_dim, hidden, config.dropout)
        self.entity_head = BoundaryHead(input_dim, hidden, config.dropout)

    @staticmethod
    def _previous(values: Tensor) -> Tensor:
        initial = torch.zeros_like(values[:, :1])
        return torch.cat((initial, values[:, :-1]), dim=1)

    def forward(
        self,
        encoded: Tensor,
        strategy: Tensor,
        delta_t: Tensor,
        action_change: Tensor,
        entity_change: Tensor,
        mask: Tensor,
        temperature: float,
    ) -> BoundaryOutput:
        temperature = max(
            float(temperature), self.config.boundary.minimum_temperature
        )

        previous = self._previous(encoded)
        difference = encoded - previous
        numeric = torch.stack(
            (
                torch.log1p(delta_t.clamp_min(0.0)),
                action_change,
                entity_change,
            ),
            dim=-1,
        ).to(encoded.dtype)

        features = torch.cat(
            (previous, encoded, difference, strategy, numeric), dim=-1
        )
        action_logit = self.action_head(features)
        entity_conditional_logit = self.entity_head(features)

        prior_scale = self.config.boundary.prior_logit_scale
        action_logit = action_logit + prior_scale * (
            action_change.to(action_logit.dtype) - 0.5
        )
        entity_conditional_logit = entity_conditional_logit + prior_scale * (
            entity_change.to(entity_conditional_logit.dtype) - 0.5
        )

        action_probability = torch.sigmoid(action_logit / temperature)
        conditional_entity = torch.sigmoid(
            entity_conditional_logit / temperature
        )

        ablation = self.config.ablation
        if not ablation.use_action_boundary:
            action_probability = action_change.to(action_probability.dtype)
        if ablation.single_boundary:
            entity_probability = action_probability
        elif not ablation.use_entity_boundary:
            entity_probability = entity_change.to(action_probability.dtype)
            entity_probability = torch.minimum(
                entity_probability, action_probability
            )
        else:
            entity_probability = action_probability * conditional_entity

        valid = mask.to(action_probability.dtype)
        action_probability = action_probability * valid
        entity_probability = entity_probability * valid

        if self.config.boundary.force_first_boundary:
            first = valid[:, :1]
            action_probability = torch.cat(
                (first, action_probability[:, 1:]), dim=1
            )
            entity_probability = torch.cat(
                (first, entity_probability[:, 1:]), dim=1
            )

        action_logit = action_logit * valid
        entity_conditional_logit = entity_conditional_logit * valid

        return BoundaryOutput(
            action_probability=action_probability,
            entity_probability=entity_probability,
            action_logit=action_logit,
            entity_conditional_logit=entity_conditional_logit,
        )
