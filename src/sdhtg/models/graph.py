from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from .config import SDHTGModelConfig


NODE_TYPES = ("status", "action", "entity")
EDGE_TYPES = (
    ("status", "temporal", "status"),
    ("status", "semantic", "status"),
    ("action", "temporal", "action"),
    ("action", "semantic", "action"),
    ("entity", "temporal", "entity"),
    ("entity", "semantic", "entity"),
    ("status", "belongs_to", "action"),
    ("action", "contains", "status"),
    ("action", "belongs_to", "entity"),
    ("entity", "contains", "action"),
)


def edge_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


@dataclass
class GraphEncoderOutput:
    graph: Batch
    node_embeddings: dict[str, Tensor]


class WeightedRelationConvolution(MessagePassing):
    """
    Sparse relation-specific message passing.

    edge_attr[:, 0] is a differentiable relation weight.
    edge_attr[:, 1] is positional distance.
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__(aggr="add", node_dim=0)
        self.source_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(target)
        messages = self.propagate(
            edge_index=edge_index,
            x=(source, target),
            edge_attr=edge_attr,
            size=(source.shape[0], target.shape[0]),
        )
        # Relation-wise weighted normalization (manuscript Eq. 44):
        # m_i = sum_j w_ji m_{j->i} / (sum_j w_ji + eps).
        structural_weight = edge_attr[:, :1].clamp_min(0.0)
        weight_sum = scatter(
            structural_weight,
            edge_index[1],
            dim=0,
            dim_size=target.shape[0],
            reduce="sum",
        )
        return messages / weight_sum.clamp_min(1e-8)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        structural_weight = edge_attr[:, :1].clamp_min(0.0)
        learned_gate = self.edge_gate(edge_attr)
        message = self.source_projection(x_j)
        return self.dropout(message * learned_gate * structural_weight)


class HeterogeneousMessageLayer(nn.Module):
    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.relations = nn.ModuleDict(
            {
                edge_key(relation): WeightedRelationConvolution(
                    hidden, config.dropout
                )
                for relation in EDGE_TYPES
            }
        )
        self.self_projection = nn.ModuleDict(
            {node_type: nn.Linear(hidden, hidden) for node_type in NODE_TYPES}
        )
        self.update = nn.ModuleDict(
            {node_type: nn.GRUCell(hidden, hidden) for node_type in NODE_TYPES}
        )
        self.norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden) for node_type in NODE_TYPES}
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        graph: Batch,
        embeddings: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        messages = {
            node_type: torch.zeros_like(embeddings[node_type])
            for node_type in NODE_TYPES
        }

        for relation in EDGE_TYPES:
            source_type, _, target_type = relation
            store = graph[relation]
            contribution = self.relations[edge_key(relation)](
                embeddings[source_type],
                embeddings[target_type],
                store.edge_index,
                store.edge_attr,
            )
            messages[target_type] = messages[target_type] + contribution

        updated = {}
        for node_type in NODE_TYPES:
            previous = embeddings[node_type]
            aggregate = (
                messages[node_type]
                + self.self_projection[node_type](previous)
            )
            next_value = self.update[node_type](aggregate, previous)
            updated[node_type] = self.norm[node_type](
                previous + self.dropout(next_value)
            )
        return updated


class HeterogeneousTemporalGraphEncoder(nn.Module):
    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            HeterogeneousMessageLayer(config)
            for _ in range(config.graph_layers)
        )

    def forward(self, graphs: list[HeteroData]) -> GraphEncoderOutput:
        if not graphs:
            raise ValueError("at least one heterogeneous graph is required")
        # P1: pre-batched graph (single entry with _num_graphs and batch vectors)
        if len(graphs) == 1:
            graph = graphs[0]
        else:
            graph = Batch.from_data_list(graphs)
        embeddings = {
            node_type: graph[node_type].x for node_type in NODE_TYPES
        }
        for layer in self.layers:
            embeddings = layer(graph, embeddings)
        return GraphEncoderOutput(graph=graph, node_embeddings=embeddings)
