from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .config import SDHTGModelConfig
from .hierarchy import HierarchyOutput, HierarchyLevel


NodeType = str
EdgeType = tuple[str, str, str]


@dataclass
class GraphBuildResult:
    graphs: list[HeteroData]
    status_counts: Tensor
    action_counts: Tensor
    entity_counts: Tensor


def _empty_edge_index(device: torch.device) -> Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _empty_edge_attr(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.empty((0, 2), dtype=dtype, device=device)


def local_temporal_edges(
    positions: Tensor,
    radius: int,
    self_loops: bool,
) -> tuple[Tensor, Tensor]:
    count = positions.numel()
    device = positions.device
    dtype = positions.dtype

    if count == 0:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)

    offsets = torch.arange(-radius, radius + 1, device=device)
    if not self_loops:
        offsets = offsets[offsets != 0]

    src = torch.arange(count, device=device).unsqueeze(1)          # [count, 1]
    tgt = src + offsets.unsqueeze(0)                               # [count, 2R+1]
    valid = (tgt >= 0) & (tgt < count)                             # [count, 2R+1]
    src = src.expand(-1, offsets.size(0))[valid]
    tgt = tgt[valid]

    if src.numel() == 0:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)

    delta = (positions[tgt] - positions[src]).abs()
    weight = torch.exp(-delta / max(float(radius), 1.0))
    edge_index = torch.stack((src, tgt), dim=0)
    edge_attr = torch.stack((weight, delta), dim=-1)
    return edge_index, edge_attr


def semantic_edges(
    semantic_ids: Tensor,
    positions: Tensor,
    maximum_neighbors: int,
) -> tuple[Tensor, Tensor]:
    count = semantic_ids.numel()
    device = semantic_ids.device
    dtype = positions.dtype

    groups: dict[int, list[int]] = {}
    for idx in range(count):
        sid = int(semantic_ids[idx].item())
        if sid > 1:
            groups.setdefault(sid, []).append(idx)

    if not groups:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)

    src_list, tgt_list, w_list, d_list = [], [], [], []

    for identifier, group in groups.items():
        m = len(group)
        if m < 2:
            continue

        # GPU distance matrix: [m, m], no .item() sync
        pos_g = positions[group].unsqueeze(1)                # [m, 1]
        dist = (pos_g - pos_g.T).abs()                       # [m, m]
        dist.fill_diagonal_(float('inf'))

        k = min(maximum_neighbors, m - 1)
        topk_val, topk_idx = dist.topk(k, dim=1, largest=False)  # [m, k]

        # Vectorised edge construction: no Python inner loops
        grp = torch.tensor(group, device=device)             # [m]
        src_idx = grp.unsqueeze(1).expand(-1, k).reshape(-1) # [m*k]
        tgt_idx = grp[topk_idx.reshape(-1)]                  # [m*k]

        src_list.append(src_idx)
        tgt_list.append(tgt_idx)
        w_list.append((1.0 / (1.0 + topk_val)).reshape(-1))
        d_list.append(topk_val.reshape(-1))

    if not src_list:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)

    return (
        torch.stack([torch.cat(src_list), torch.cat(tgt_list)], dim=0),
        torch.stack([torch.cat(w_list), torch.cat(d_list)], dim=-1),
    )


def containment_edges(
    membership: Tensor,
    source_count: int,
    target_count: int,
    source_positions: Tensor,
    target_positions: Tensor,
    minimum_weight: float,
) -> tuple[Tensor, Tensor]:
    """
    membership has shape [source candidates, target candidates].
    The edge weight remains attached to the autograd graph.
    """
    device = membership.device
    dtype = membership.dtype
    if source_count == 0 or target_count == 0:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)

    weights = membership[:source_count, :target_count]
    indices = torch.nonzero(weights > minimum_weight, as_tuple=False)
    if indices.numel() == 0:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)

    source = indices[:, 0]
    target = indices[:, 1]
    selected = weights[source, target]
    delta = (
        source_positions[source] - target_positions[target]
    ).abs()
    edge_index = torch.stack((source, target), dim=0)
    edge_attr = torch.stack((selected, delta), dim=-1)
    return edge_index, edge_attr


class HeterogeneousGraphBuilder:
    def __init__(self, config: SDHTGModelConfig):
        self.config = config

    @staticmethod
    def _count(level: HierarchyLevel, index: int) -> int:
        return int(level.mask[index].long().sum().item())

    @staticmethod
    def _assign_nodes(
        graph: HeteroData,
        node_type: str,
        level: HierarchyLevel,
        batch_index: int,
        count: int,
    ) -> None:
        graph[node_type].x = level.features[batch_index, :count]
        graph[node_type].mass = level.mass[batch_index, :count]
        graph[node_type].position = level.positions[batch_index, :count]
        graph[node_type].semantic_id = level.semantic_id[
            batch_index, :count
        ]

    @staticmethod
    def _assign_edges(
        graph: HeteroData,
        edge_type: EdgeType,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> None:
        graph[edge_type].edge_index = edge_index
        graph[edge_type].edge_attr = edge_attr

    def _within_level_edges(
        self,
        graph: HeteroData,
        node_type: str,
        level: HierarchyLevel,
        batch_index: int,
        count: int,
    ) -> None:
        positions = level.positions[batch_index, :count]
        semantic_ids = level.semantic_id[batch_index, :count]
        relations = self.config.graph_relations
        ablation = self.config.ablation

        if relations.temporal and ablation.use_temporal_edges:
            edge_index, edge_attr = local_temporal_edges(
                positions,
                radius=int(
                    self.config.local_temporal_radius[node_type]
                ),
                self_loops=relations.self_loops,
            )
        else:
            edge_index = _empty_edge_index(positions.device)
            edge_attr = _empty_edge_attr(
                positions.device, positions.dtype
            )
        self._assign_edges(
            graph,
            (node_type, "temporal", node_type),
            edge_index,
            edge_attr,
        )

        if relations.semantic and ablation.use_semantic_edges:
            edge_index, edge_attr = semantic_edges(
                semantic_ids,
                positions,
                maximum_neighbors=int(
                    self.config.semantic_neighbors[node_type]
                ),
            )
        else:
            edge_index = _empty_edge_index(positions.device)
            edge_attr = _empty_edge_attr(
                positions.device, positions.dtype
            )
        self._assign_edges(
            graph,
            (node_type, "semantic", node_type),
            edge_index,
            edge_attr,
        )

    def build(self, hierarchy: HierarchyOutput) -> GraphBuildResult:
        batch_size = hierarchy.status.features.shape[0]
        graphs = []
        status_counts = []
        action_counts = []
        entity_counts = []

        for batch_index in range(batch_size):
            graph = HeteroData()

            status_count = self._count(hierarchy.status, batch_index)
            action_count = self._count(hierarchy.action, batch_index)
            entity_count = self._count(hierarchy.entity, batch_index)

            status_counts.append(status_count)
            action_counts.append(action_count)
            entity_counts.append(entity_count)

            self._assign_nodes(
                graph, "status", hierarchy.status, batch_index, status_count
            )
            self._assign_nodes(
                graph, "action", hierarchy.action, batch_index, action_count
            )
            self._assign_nodes(
                graph, "entity", hierarchy.entity, batch_index, entity_count
            )

            self._within_level_edges(
                graph, "status", hierarchy.status, batch_index, status_count
            )
            self._within_level_edges(
                graph, "action", hierarchy.action, batch_index, action_count
            )
            self._within_level_edges(
                graph, "entity", hierarchy.entity, batch_index, entity_count
            )

            minimum = self.config.hierarchy.minimum_node_mass
            status_action_index, status_action_attr = containment_edges(
                hierarchy.status_to_action[batch_index],
                status_count,
                action_count,
                hierarchy.status.positions[batch_index, :status_count],
                hierarchy.action.positions[batch_index, :action_count],
                minimum,
            )
            action_entity_index, action_entity_attr = containment_edges(
                hierarchy.action_to_entity[batch_index],
                action_count,
                entity_count,
                hierarchy.action.positions[batch_index, :action_count],
                hierarchy.entity.positions[batch_index, :entity_count],
                minimum,
            )

            if (
                not self.config.graph_relations.containment
                or not self.config.ablation.use_cross_level_messages
            ):
                status_action_index = _empty_edge_index(
                    hierarchy.status.features.device
                )
                status_action_attr = _empty_edge_attr(
                    hierarchy.status.features.device,
                    hierarchy.status.features.dtype,
                )
                action_entity_index = _empty_edge_index(
                    hierarchy.status.features.device
                )
                action_entity_attr = _empty_edge_attr(
                    hierarchy.status.features.device,
                    hierarchy.status.features.dtype,
                )

            if self.config.ablation.detach_boundary_from_graph:
                status_action_attr = status_action_attr.detach()
                action_entity_attr = action_entity_attr.detach()

            self._assign_edges(
                graph,
                ("status", "belongs_to", "action"),
                status_action_index,
                status_action_attr,
            )
            self._assign_edges(
                graph,
                ("action", "belongs_to", "entity"),
                action_entity_index,
                action_entity_attr,
            )

            if (
                self.config.graph_relations.reverse_containment
                and self.config.ablation.use_cross_level_messages
            ):
                self._assign_edges(
                    graph,
                    ("action", "contains", "status"),
                    status_action_index.flip(0),
                    status_action_attr,
                )
                self._assign_edges(
                    graph,
                    ("entity", "contains", "action"),
                    action_entity_index.flip(0),
                    action_entity_attr,
                )
            else:
                device = hierarchy.status.features.device
                dtype = hierarchy.status.features.dtype
                self._assign_edges(
                    graph,
                    ("action", "contains", "status"),
                    _empty_edge_index(device),
                    _empty_edge_attr(device, dtype),
                )
                self._assign_edges(
                    graph,
                    ("entity", "contains", "action"),
                    _empty_edge_index(device),
                    _empty_edge_attr(device, dtype),
                )

            graph.sample_index = torch.tensor(
                [batch_index],
                dtype=torch.long,
                device=hierarchy.status.features.device,
            )
            graphs.append(graph)

        device = hierarchy.status.features.device
        return GraphBuildResult(
            graphs=graphs,
            status_counts=torch.tensor(
                status_counts, dtype=torch.long, device=device
            ),
            action_counts=torch.tensor(
                action_counts, dtype=torch.long, device=device
            ),
            entity_counts=torch.tensor(
                entity_counts, dtype=torch.long, device=device
            ),
        )
