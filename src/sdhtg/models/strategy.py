from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import SDHTGModelConfig
from .utils import zero_padding


@dataclass
class StrategyOutput:
    modulated: Tensor
    per_event_strategy: Tensor
    sequence_strategy: Tensor
    gamma: Tensor
    beta: Tensor


class CausalStrategyFiLM(nn.Module):
    """
    Generates an event-wise strategy from the causal event representation.

    Unlike a global mean over the complete sequence, event strategy at time t
    cannot access a future event. The final valid strategy is separately exposed
    as sequence_strategy for graph-level evidence fusion.
    """

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        strategy_dim = config.strategy_dim

        self.context_rnn = nn.GRU(
            input_size=hidden + 3,
            hidden_size=strategy_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.strategy_norm = nn.LayerNorm(strategy_dim)
        self.gamma = nn.Linear(strategy_dim, hidden)
        self.beta = nn.Linear(strategy_dim, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(config.dropout)

        # Identity FiLM at initialization.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(
        self,
        encoded: Tensor,
        delta_t: Tensor,
        action_change: Tensor,
        entity_change: Tensor,
        mask: Tensor,
        strength: float,
        enabled: bool = True,
    ) -> StrategyOutput:
        if strength < 0:
            raise ValueError("FiLM strength must be non-negative")

        numeric = torch.stack(
            (
                torch.log1p(delta_t.clamp_min(0.0)),
                action_change,
                entity_change,
            ),
            dim=-1,
        ).to(encoded.dtype)
        numeric = numeric * mask.unsqueeze(-1).to(encoded.dtype)
        context_input = torch.cat((encoded, numeric), dim=-1)
        strategy, _ = self.context_rnn(context_input)
        strategy = self.strategy_norm(strategy)
        strategy = zero_padding(strategy, mask)

        if enabled:
            gamma = 1.0 + strength * torch.tanh(self.gamma(strategy))
            beta = strength * torch.tanh(self.beta(strategy))
            modulated = self.output_norm(gamma * encoded + beta)
        else:
            gamma = torch.ones_like(encoded)
            beta = torch.zeros_like(encoded)
            modulated = encoded

        modulated = zero_padding(self.dropout(modulated), mask)
        gamma = gamma * mask.unsqueeze(-1).to(gamma.dtype)
        beta = beta * mask.unsqueeze(-1).to(beta.dtype)

        lengths = mask.long().sum(dim=1)
        last_index = (lengths - 1).clamp_min(0)
        batch_index = torch.arange(encoded.shape[0], device=encoded.device)
        sequence_strategy = strategy[batch_index, last_index]

        return StrategyOutput(
            modulated=modulated,
            per_event_strategy=strategy,
            sequence_strategy=sequence_strategy,
            gamma=gamma,
            beta=beta,
        )
