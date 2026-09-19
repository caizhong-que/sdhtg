from __future__ import annotations

"""Unified-input-protocol baselines.

All three baselines consume exactly the same event features as SDHTG
(template / entity / action / status / continuous time, gated fusion) and
share the same prototype detector, so differences come from the temporal or
graph encoder rather than from the input representation.

    TCNBaseline          dilated causal convolutions over fused event features
    TransformerBaseline  encoder-only Transformer with padding masks
    FlatGraphBaseline    single-level event graph (temporal + semantic edges),
                         no learned boundaries and no hierarchy
"""

import math

import torch
from torch import Tensor, nn

from .config import SDHTGModelConfig
from .detector import _prototype_metrics
from .event_encoder import (
    ContinuousTimeEncoding,
    GatedMultiSourceFusion,
)
from .graph import WeightedRelationConvolution
from .graph_builder_batched import local_temporal_edges, semantic_edges
from .sdhtg import SDHTGOutput
from .utils import masked_mean, zero_padding


class BaselineEventFeatures(nn.Module):
    """Template/entity/action/status/time embeddings + gated fusion (no GRU)."""

    SOURCE_NAMES = ("template", "entity", "action", "status", "time")

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
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
        self.fusion = GatedMultiSourceFusion(
            hidden_dim=hidden,
            source_count=len(self.SOURCE_NAMES),
            dropout=config.dropout,
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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        mask = batch["mask"]
        dtype = self.template_embedding.weight.dtype
        time_features = self.time_encoder(batch["delta_t"].to(dtype=dtype), mask)
        sources = [
            self.template_embedding(batch["template_id"]),
            self.entity_embedding(batch["entity_id"]),
            self.action_embedding(batch["action_id"]),
            self.status_embedding(batch["status_id"]),
            self.time_projection(time_features),
        ]
        sources = [zero_padding(source, mask) for source in sources]
        fused, gates = self.fusion(sources, mask)
        return self.output_norm(fused), gates


def _prototype_head(
    config: SDHTGModelConfig,
    embedding: Tensor,
    prototypes: nn.Parameter,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    distance, nearest, diversity, distances = _prototype_metrics(
        embedding,
        prototypes,
        config.prototype_temperature,
        config.prototype_similarity_threshold,
    )
    return distance, nearest, diversity, distances


class TCNBaseline(nn.Module):
    """Dilated causal temporal convolution baseline."""

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.features = BaselineEventFeatures(config)
        self.blocks = nn.ModuleList(
            _CausalConvBlock(hidden, dilation=dilation, dropout=config.dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.embedding_projection = nn.Linear(hidden * 2, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.normal_prototypes = nn.Parameter(
            torch.empty(config.num_normal_prototypes, hidden)
        )
        nn.init.normal_(self.normal_prototypes, mean=0.0, std=0.02)
        self.prototype_scale_logit = nn.Parameter(torch.zeros(()))
        self.prototype_center = nn.Parameter(torch.zeros(()))

    def forward(self, batch, **_kwargs) -> SDHTGOutput:
        mask = batch["mask"]
        values, gates = self.features(batch)
        for block in self.blocks:
            values = block(values, mask)
        pooled = torch.cat(
            (masked_mean(values, mask, dim=1), _masked_max(values, mask)), dim=-1
        )
        embedding = self.embedding_projection(pooled)
        logit = self.head(embedding).squeeze(-1)
        distance, nearest, diversity, distances = _prototype_head(
            self.config, embedding, self.normal_prototypes
        )
        if self.config.ablation.use_prototypes:
            scale = torch.nn.functional.softplus(self.prototype_scale_logit)
            anomaly_logit = logit + scale * (distance - self.prototype_center)
        else:
            anomaly_logit = logit
        return _output(
            self.config, anomaly_logit, logit, embedding, distances, distance,
            nearest, diversity, gates, batch,
        )


class _CausalConvBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.kernel = 3
        self.conv = nn.Conv1d(
            hidden, hidden, kernel_size=self.kernel, dilation=dilation
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        padded = values.transpose(1, 2)
        padding = (self.kernel - 1) * self.dilation
        padded = torch.nn.functional.pad(padded, (padding, 0))
        convolved = self.conv(padded).transpose(1, 2)
        updated = self.norm(values + self.dropout(self.activation(convolved)))
        return zero_padding(updated, mask)


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    minimum = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask.unsqueeze(-1), minimum)
    result = masked.max(dim=1).values
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


class TransformerBaseline(nn.Module):
    """Encoder-only Transformer with padding masks."""

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.features = BaselineEventFeatures(config)
        self.position_projection = nn.Linear(1, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.graph_heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.graph_layers)
        self.embedding_projection = nn.Linear(hidden * 2, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.normal_prototypes = nn.Parameter(
            torch.empty(config.num_normal_prototypes, hidden)
        )
        nn.init.normal_(self.normal_prototypes, mean=0.0, std=0.02)
        self.prototype_scale_logit = nn.Parameter(torch.zeros(()))
        self.prototype_center = nn.Parameter(torch.zeros(()))

    def forward(self, batch, **_kwargs) -> SDHTGOutput:
        mask = batch["mask"]
        values, gates = self.features(batch)
        steps = values.shape[1]
        positions = torch.arange(steps, device=values.device, dtype=values.dtype)
        positions = (positions / max(steps - 1, 1)).view(1, steps, 1)
        positions = positions.expand(values.shape[0], steps, 1)
        values = values + self.position_projection(positions)
        encoded = self.encoder(
            values, src_key_padding_mask=~mask
        )
        encoded = zero_padding(encoded, mask)
        pooled = torch.cat(
            (masked_mean(encoded, mask, dim=1), _masked_max(encoded, mask)), dim=-1
        )
        embedding = self.embedding_projection(pooled)
        logit = self.head(embedding).squeeze(-1)
        distance, nearest, diversity, distances = _prototype_head(
            self.config, embedding, self.normal_prototypes
        )
        if self.config.ablation.use_prototypes:
            scale = torch.nn.functional.softplus(self.prototype_scale_logit)
            anomaly_logit = logit + scale * (distance - self.prototype_center)
        else:
            anomaly_logit = logit
        return _output(
            self.config, anomaly_logit, logit, embedding, distances, distance,
            nearest, diversity, gates, batch,
        )


class FlatGraphMessageLayer(nn.Module):
    """Edge-weight-aware message passing on a single node type."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.temporal = WeightedRelationConvolution(hidden, dropout)
        self.semantic = WeightedRelationConvolution(hidden, dropout)
        self.self_projection = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, temporal, semantic) -> Tensor:
        messages = self.temporal(values, values, *temporal)
        messages = messages + self.semantic(values, values, *semantic)
        aggregate = messages + self.self_projection(values)
        updated = self.update(aggregate, values)
        return self.norm(values + self.dropout(updated))


class FlatGraphBaseline(nn.Module):
    """Single-level event graph (no learned boundaries, no hierarchy)."""

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.features = BaselineEventFeatures(config)
        self.layers = nn.ModuleList(
            FlatGraphMessageLayer(hidden, config.dropout)
            for _ in range(config.graph_layers)
        )
        self.embedding_projection = nn.Linear(hidden * 2, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.normal_prototypes = nn.Parameter(
            torch.empty(config.num_normal_prototypes, hidden)
        )
        nn.init.normal_(self.normal_prototypes, mean=0.0, std=0.02)
        self.prototype_scale_logit = nn.Parameter(torch.zeros(()))
        self.prototype_center = nn.Parameter(torch.zeros(()))

    def _edges(self, batch) -> tuple[tuple, tuple]:
        mask = batch["mask"]
        temporal, semantic = [], []
        radius = self.config.local_temporal_radius["status"]
        neighbors = self.config.semantic_neighbors["status"]
        for index in range(mask.shape[0]):
            length = int(mask[index].sum().item())
            positions = torch.arange(
                length, device=mask.device, dtype=torch.float32
            )
            temporal.append(
                local_temporal_edges(
                    positions,
                    radius,
                    self.config.graph_relations.self_loops,
                )
            )
            semantic_ids = batch["entity_id"][index, :length]
            semantic.append(semantic_edges(semantic_ids, positions, neighbors))
        return temporal, semantic

    def forward(self, batch, **_kwargs) -> SDHTGOutput:
        mask = batch["mask"]
        values, gates = self.features(batch)
        temporal_edges, semantic_edges_list = self._edges(batch)
        edge_total = sum(
            int(edge_index.shape[1])
            for group in (temporal_edges, semantic_edges_list)
            for edge_index, _ in group
        )
        # Encode each sample separately so edge indices stay local.
        encoded = []
        for index in range(mask.shape[0]):
            length = int(mask[index].sum().item())
            sample = values[index, :length]
            for layer in self.layers:
                sample = layer(
                    sample, temporal_edges[index], semantic_edges_list[index]
                )
            padded = values.new_zeros(values.shape[1], values.shape[-1])
            padded[:length] = sample
            encoded.append(padded)
        encoded = torch.stack(encoded, dim=0)
        pooled = torch.cat(
            (masked_mean(encoded, mask, dim=1), _masked_max(encoded, mask)), dim=-1
        )
        embedding = self.embedding_projection(pooled)
        logit = self.head(embedding).squeeze(-1)
        distance, nearest, diversity, distances = _prototype_head(
            self.config, embedding, self.normal_prototypes
        )
        if self.config.ablation.use_prototypes:
            scale = torch.nn.functional.softplus(self.prototype_scale_logit)
            anomaly_logit = logit + scale * (distance - self.prototype_center)
        else:
            anomaly_logit = logit
        return _output(
            self.config, anomaly_logit, logit, embedding, distances, distance,
            nearest, diversity, gates, batch, graph_edge_count=edge_total,
        )


def _output(
    config: SDHTGModelConfig,
    anomaly_logit: Tensor,
    level_logit: Tensor,
    pooled: Tensor,
    prototype_distances: Tensor,
    prototype_distance: Tensor,
    nearest: Tensor,
    diversity: Tensor,
    gates: Tensor,
    batch: dict[str, Tensor],
    graph_edge_count: int = 0,
) -> SDHTGOutput:
    """Package baseline outputs in the SDHTG output convention."""
    mask = batch["mask"]
    batch_size, steps = mask.shape
    device, dtype = pooled.device, pooled.dtype
    zero_scalar = anomaly_logit.sum() * 0.0
    zero_boundary = torch.zeros(batch_size, steps, device=device, dtype=dtype)
    empty_matrix = torch.zeros(batch_size, steps, 0, device=device, dtype=dtype)
    level_logits = level_logit.unsqueeze(1).expand(-1, 3)
    level_weights = torch.full(
        (batch_size, 3), 1.0 / 3.0, device=device, dtype=dtype
    )
    return SDHTGOutput(
        anomaly_logit=anomaly_logit,
        anomaly_probability=torch.sigmoid(anomaly_logit),
        level_logits=level_logits,
        level_weights=level_weights,
        graph_embedding=pooled,
        prototype_distance=prototype_distance,
        prototype_distances=prototype_distances,
        nearest_prototype=nearest,
        prototype_diversity=diversity,
        action_boundary=zero_boundary,
        entity_boundary=zero_boundary,
        action_boundary_logit=zero_boundary,
        entity_conditional_logit=zero_boundary,
        event_encoding=pooled.unsqueeze(1).expand(-1, steps, -1),
        modulated_event_encoding=pooled.unsqueeze(1).expand(-1, steps, -1),
        source_gate=gates,
        event_strategy=zero_scalar,
        sequence_strategy=zero_scalar,
        status_to_action=empty_matrix,
        action_to_entity=empty_matrix,
        status_node_mask=mask,
        action_node_mask=torch.zeros_like(mask),
        entity_node_mask=torch.zeros_like(mask),
        graph_batch=None,
        graph_edge_count=graph_edge_count,
    )
