from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_mean_pool

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
    nearest_prototype: Tensor


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

    @staticmethod
    def _number_of_graphs(output: GraphEncoderOutput) -> int:
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
        level_logits = []
        for graph_index in range(graph_count):
            selected = node_logits[batch == graph_index]
            if selected.numel() == 0:
                level_logits.append(node_logits.new_zeros(()))
            else:
                level_logits.append(
                    temperature
                    * torch.logsumexp(selected / temperature, dim=0)
                )
        return embedding, torch.stack(level_logits)

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

        normalized_embedding = torch.nn.functional.normalize(
            graph_embedding, dim=-1
        )
        normalized_prototypes = torch.nn.functional.normalize(
            self.normal_prototypes, dim=-1
        )
        distances = 1.0 - normalized_embedding @ normalized_prototypes.T
        prototype_distance, nearest = distances.min(dim=1)

        if self.config.ablation.use_prototypes:
            # A positive learnable-free monotonic term keeps the prototype score
            # interpretable and allows the prototype loss to shape embeddings.
            anomaly_logit = weighted_logit + prototype_distance
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
            nearest_prototype=nearest,
        )
