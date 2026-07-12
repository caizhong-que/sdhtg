from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import SDHTGModelConfig
from .utils import lengths_from_mask, validate_sequence_batch, zero_padding


@dataclass
class EventEncoderOutput:
    encoded: Tensor
    fused_inputs: Tensor
    source_gate: Tensor
    time_features: Tensor


class ContinuousTimeEncoding(nn.Module):
    """Causal encoding of the elapsed time preceding each event."""

    def __init__(self, output_dim: int):
        super().__init__()
        if output_dim < 4:
            raise ValueError("time encoding dimension must be at least four")
        periodic_dim = output_dim - 2
        self.log_frequencies = nn.Parameter(
            torch.linspace(-4.0, 2.0, periodic_dim)
        )
        self.projection = nn.Sequential(
            nn.Linear(periodic_dim + 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, delta_t: Tensor, mask: Tensor) -> Tensor:
        delta = torch.log1p(delta_t.clamp_min(0.0))
        frequencies = self.log_frequencies.exp()
        phase = delta.unsqueeze(-1) * frequencies
        periodic = torch.sin(phase)
        raw = torch.cat(
            (
                delta.unsqueeze(-1),
                torch.sqrt(delta + 1e-8).unsqueeze(-1),
                periodic,
            ),
            dim=-1,
        )
        return self.projection(raw) * mask.unsqueeze(-1).to(raw.dtype)


class GatedMultiSourceFusion(nn.Module):
    """
    Learns a convex source mixture for each event.

    Sources remain separately identifiable, which permits source ablations and
    source-gate diagnostics. A shared projection maps all sources to hidden_dim.
    """

    def __init__(self, hidden_dim: int, source_count: int, dropout: float):
        super().__init__()
        self.source_count = source_count
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * source_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, source_count),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, sources: list[Tensor], mask: Tensor) -> tuple[Tensor, Tensor]:
        if len(sources) != self.source_count:
            raise ValueError(
                f"expected {self.source_count} sources, got {len(sources)}"
            )
        stacked = torch.stack(sources, dim=2)
        concatenated = torch.cat(sources, dim=-1)
        gates = torch.softmax(self.gate(concatenated), dim=-1)
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=2)
        fused = self.output(fused)
        fused = zero_padding(fused, mask)
        gates = gates * mask.unsqueeze(-1).to(gates.dtype)
        return fused, gates


class MultiSourceEventEncoder(nn.Module):
    """
    Template, entity, action, status, and elapsed-time event encoder.

    The recurrent encoder is unidirectional. Consequently, encoded[:, t] depends
    only on events at positions <= t. Packed sequences prevent padded positions
    from affecting the recurrent state.
    """

    SOURCE_NAMES = ("template", "entity", "action", "status", "time")

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.template_embedding = nn.Embedding(
            config.template_vocab_size, hidden, padding_idx=0
        )
        self.entity_embedding = nn.Embedding(
            config.entity_vocab_size, hidden, padding_idx=0
        )
        self.action_embedding = nn.Embedding(
            config.action_vocab_size, hidden, padding_idx=0
        )
        self.status_embedding = nn.Embedding(
            config.status_vocab_size, hidden, padding_idx=0
        )
        self.time_encoder = ContinuousTimeEncoding(config.time_dim)
        self.time_projection = nn.Linear(config.time_dim, hidden)

        self.source_scale = nn.Parameter(torch.ones(len(self.SOURCE_NAMES)))
        self.fusion = GatedMultiSourceFusion(
            hidden_dim=hidden,
            source_count=len(self.SOURCE_NAMES),
            dropout=config.dropout,
        )
        self.encoder = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=config.num_gru_layers,
            dropout=config.dropout if config.num_gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.output_norm = nn.LayerNorm(hidden)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.template_embedding,
            self.entity_embedding,
            self.action_embedding,
            self.status_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()

    def _source_enabled(self, name: str) -> bool:
        return {
            "template": self.config.ablation.use_template_source,
            "entity": self.config.ablation.use_entity_source,
            "action": self.config.ablation.use_action_source,
            "status": self.config.ablation.use_status_source,
            "time": self.config.ablation.use_time_source,
        }[name]

    def forward(self, batch: dict[str, Tensor]) -> EventEncoderOutput:
        _, steps = validate_sequence_batch(batch)
        mask = batch["mask"]
        dtype = self.template_embedding.weight.dtype

        time_features = self.time_encoder(
            batch["delta_t"].to(dtype=dtype), mask
        )
        raw_sources = {
            "template": self.template_embedding(batch["template_id"]),
            "entity": self.entity_embedding(batch["entity_id"]),
            "action": self.action_embedding(batch["action_id"]),
            "status": self.status_embedding(batch["status_id"]),
            "time": self.time_projection(time_features),
        }

        sources = []
        for index, name in enumerate(self.SOURCE_NAMES):
            source = raw_sources[name]
            if not self._source_enabled(name):
                source = torch.zeros_like(source)
            source = source * self.source_scale[index]
            sources.append(zero_padding(source, mask))

        fused, source_gate = self.fusion(sources, mask)

        lengths = lengths_from_mask(mask).cpu()
        packed = pack_padded_sequence(
            fused,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=steps,
        )
        encoded = self.output_norm(encoded)
        encoded = zero_padding(encoded, mask)

        return EventEncoderOutput(
            encoded=encoded,
            fused_inputs=fused,
            source_gate=source_gate,
            time_features=time_features,
        )
