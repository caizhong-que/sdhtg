from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

from .config import SDHTGModelConfig
from .graph import GraphEncoderOutput, NODE_TYPES


@dataclass
class DetectorOutput:
    anomaly_logit: Tensor
    anomaly_probability: Tensor
    level_logits: Tensor
    level_weights: Tensor
    level_embeddings: Tensor
    graph_embedding: Tensor
    prototype_distance: Tensor
    prototype_distances: Tensor
    nearest_prototype: Tensor
    prototype_diversity: Tensor


def _prototype_metrics(
    embedding: Tensor,
    prototypes: Tensor,
    temperature: float,
    similarity_threshold: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    normalized_embedding = torch.nn.functional.normalize(embedding, dim=-1)
    normalized_prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    distances = 1.0 - normalized_embedding @ normalized_prototypes.T

    num_prototypes = normalized_prototypes.shape[0]
    if temperature > 0.0 and num_prototypes >= 1:
        # Soft-min over prototypes (manuscript Eq. 53).
        prototype_distance = -temperature * torch.logsumexp(
            -distances / temperature, dim=1
        ) + temperature * math.log(num_prototypes)
    else:
        prototype_distance, _ = distances.min(dim=1)
    nearest = distances.argmin(dim=1)

    if num_prototypes >= 2:
        eye = torch.eye(
            num_prototypes, device=normalized_prototypes.device
        )
        pairwise = normalized_prototypes @ normalized_prototypes.T
        off_diagonal = pairwise * (1.0 - eye)
        # Manuscript Eq. 62: hinge diversity on prototype similarity.
        prototype_diversity = (
            torch.relu(off_diagonal - similarity_threshold).sum()
            / (num_prototypes * (num_prototypes - 1))
        )
    else:
        prototype_diversity = torch.zeros(
            (), device=normalized_prototypes.device
        )
    return prototype_distance, nearest, prototype_diversity, distances


class LevelDetectionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values).squeeze(-1)


class HierarchicalAnomalyDetector(nn.Module):
    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config

        self.level_heads = nn.ModuleDict(
            {
                node_type: LevelDetectionHead(hidden, config.dropout)
                for node_type in NODE_TYPES
            }
        )
        self.level_gate = nn.Sequential(
            nn.Linear(config.strategy_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, len(NODE_TYPES)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * len(NODE_TYPES), hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
        )

        self.normal_prototypes = nn.Parameter(
            torch.empty(config.num_normal_prototypes, hidden)
        )
        nn.init.normal_(self.normal_prototypes, mean=0.0, std=0.02)
        # Learnable prototype scale and distance center (manuscript Eq. 54).
        self.prototype_scale_logit = nn.Parameter(torch.zeros(()))
        self.prototype_center = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _number_of_graphs(output: GraphEncoderOutput) -> int:
        # Pre-batched graph (P1) sets _num_graphs; fallback to batch vector
        for node_type in NODE_TYPES:
            b = output.graph[node_type].batch
            if b.numel() > 0:
                return int(b.max().item()) + 1
        return int(output.graph.num_graphs)

    def _pool_level(
        self,
        output: GraphEncoderOutput,
        node_type: str,
    ) -> tuple[Tensor, Tensor]:
        values = output.node_embeddings[node_type]
        batch = output.graph[node_type].batch
        graph_count = self._number_of_graphs(output)

        embedding = global_mean_pool(
            values, batch, size=graph_count
        )
        node_logits = self.level_heads[node_type](values)

        temperature = self.config.detector_pool_temperature
        scaled = node_logits / temperature
        maximum = scatter(
            scaled,
            batch,
            dim=0,
            dim_size=graph_count,
            reduce="max",
        )
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        exp_sum = scatter(
            torch.exp(scaled - maximum[batch]),
            batch,
            dim=0,
            dim_size=graph_count,
            reduce="sum",
        )
        counts = scatter(
            torch.ones_like(node_logits),
            batch,
            dim=0,
            dim_size=graph_count,
            reduce="sum",
        )
        # Node-count-normalized smoothed max pooling (manuscript Eq. 48).
        logsumexp = maximum + torch.log(exp_sum.clamp_min(1e-12))
        level_logits = temperature * (
            logsumexp - torch.log(counts.clamp_min(1.0))
        )
        level_logits = torch.where(
            counts > 0, level_logits, torch.zeros_like(level_logits)
        )
        return embedding, level_logits

    def forward(
        self,
        graph_output: GraphEncoderOutput,
        sequence_strategy: Tensor,
    ) -> DetectorOutput:
        embeddings = []
        logits = []
        for node_type in NODE_TYPES:
            level_embedding, level_logit = self._pool_level(
                graph_output, node_type
            )
            embeddings.append(level_embedding)
            logits.append(level_logit)

        level_embeddings = torch.stack(embeddings, dim=1)
        level_logits = torch.stack(logits, dim=1)
        level_weights = torch.softmax(
            self.level_gate(sequence_strategy), dim=-1
        )

        weighted_logit = (level_weights * level_logits).sum(dim=-1)
        graph_embedding = self.fusion(torch.cat(embeddings, dim=-1))

        prototype_distance, nearest, prototype_diversity, prototype_distances = (
            _prototype_metrics(
            graph_embedding,
            self.normal_prototypes,
            self.config.prototype_temperature,
                self.config.prototype_similarity_threshold,
            )
        )

        if self.config.ablation.use_prototypes:
            if self.config.prototype_scale_override is None:
                scale = torch.nn.functional.softplus(self.prototype_scale_logit)
            else:
                # Fixed lambda_p for the sensitivity sweep (manuscript 6.8).
                scale = self.prototype_scale_logit.new_tensor(
                    float(self.config.prototype_scale_override)
                )
            anomaly_logit = weighted_logit + scale * (
                prototype_distance - self.prototype_center
            )
        else:
            anomaly_logit = weighted_logit

        return DetectorOutput(
            anomaly_logit=anomaly_logit,
            anomaly_probability=torch.sigmoid(anomaly_logit),
            level_logits=level_logits,
            level_weights=level_weights,
            level_embeddings=level_embeddings,
            graph_embedding=graph_embedding,
            prototype_distance=prototype_distance,
            prototype_distances=prototype_distances,
            nearest_prototype=nearest,
            prototype_diversity=prototype_diversity,
        )


class FlatGRUDetector(nn.Module):
    """GRU-only detector: pools event features and applies a single head.

    Used when ``ablation.use_hierarchy`` is disabled, so the structural
    ablation table has a genuine non-hierarchical baseline.
    """

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
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

    def forward(
        self,
        pooled_embedding: Tensor,
        sequence_strategy: Tensor,
    ) -> DetectorOutput:
        batch = pooled_embedding.shape[0]
        logit = self.head(pooled_embedding).squeeze(-1)

        prototype_distance, nearest, prototype_diversity, prototype_distances = (
            _prototype_metrics(
            pooled_embedding,
            self.normal_prototypes,
            self.config.prototype_temperature,
                self.config.prototype_similarity_threshold,
            )
        )
        if self.config.ablation.use_prototypes:
            scale = torch.nn.functional.softplus(self.prototype_scale_logit)
            anomaly_logit = logit + scale * (
                prototype_distance - self.prototype_center
            )
        else:
            anomaly_logit = logit

        level_logits = logit.unsqueeze(1).expand(-1, len(NODE_TYPES))
        level_weights = torch.full(
            (batch, len(NODE_TYPES)),
            1.0 / len(NODE_TYPES),
            device=pooled_embedding.device,
        )
        level_embeddings = pooled_embedding.unsqueeze(1).expand(
            -1, len(NODE_TYPES), -1
        )
        return DetectorOutput(
            anomaly_logit=anomaly_logit,
            anomaly_probability=torch.sigmoid(anomaly_logit),
            level_logits=level_logits,
            level_weights=level_weights,
            level_embeddings=level_embeddings,
            graph_embedding=pooled_embedding,
            prototype_distance=prototype_distance,
            prototype_distances=prototype_distances,
            nearest_prototype=nearest,
            prototype_diversity=prototype_diversity,
        )
