from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .config import SDHTGModelConfig
from .hierarchy import HierarchyOutput, HierarchyLevel


ALL_EDGE_TYPES = (
    ("status", "temporal", "status"),
    ("status", "semantic", "status"),
    ("action", "temporal", "action"),
    ("action", "semantic", "action"),
    ("entity", "temporal", "entity"),
    ("entity", "semantic", "entity"),
    ("status", "belongs_to", "action"),
    ("action", "belongs_to", "entity"),
    ("action", "contains", "status"),
    ("entity", "contains", "action"),
)


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


def _append_edge(
    sources: list[int],
    targets: list[int],
    attributes: list[list[Tensor]],
    source: int,
    target: int,
    weight: Tensor,
    delta: Tensor,
) -> None:
    sources.append(source)
    targets.append(target)
    attributes.append([weight, delta])


def local_temporal_edges(
    positions: Tensor,
    radius: int,
    self_loops: bool,
) -> tuple[Tensor, Tensor]:
    count = positions.numel()
    device = positions.device
    dtype = positions.dtype
    sources: list[int] = []
    targets: list[int] = []
    attrs: list[list[Tensor]] = []

    for source in range(count):
        left = max(0, source - radius)
        right = min(count, source + radius + 1)
        for target in range(left, right):
            if source == target and not self_loops:
                continue
            delta = (positions[target] - positions[source]).abs()
            weight = torch.exp(-delta / max(float(radius), 1.0))
            _append_edge(
                sources, targets, attrs, source, target, weight, delta
            )

    if not sources:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    edge_index = torch.tensor(
        [sources, targets], dtype=torch.long, device=device
    )
    edge_attr = torch.stack(
        [torch.stack(pair) for pair in attrs], dim=0
    ).to(dtype)
    return edge_index, edge_attr


def semantic_edges(
    semantic_ids: Tensor,
    positions: Tensor,
    maximum_neighbors: int,
) -> tuple[Tensor, Tensor]:
    count = semantic_ids.numel()
    device = semantic_ids.device
    dtype = positions.dtype

    # Build semantic_id -> indices map: O(S) instead of O(SÂ²)
    groups: dict[int, list[int]] = {}
    for idx in range(count):
        sid = int(semantic_ids[idx].item())
        if sid > 1:
            groups.setdefault(sid, []).append(idx)

    sources: list[int] = []
    targets: list[int] = []
    attrs: list[list[Tensor]] = []

    for identifier, group in groups.items():
        group_size = len(group)
        if group_size < 2:
            continue
        # Each node connects to its nearest neighbours within the same
        # semantic group, up to maximum_neighbors.
        candidates = []
        for src_idx in group:
            src_pos = positions[src_idx]
            for tgt_idx in group:
                if tgt_idx == src_idx:
                    continue
                dist = float((positions[tgt_idx] - src_pos).abs().item())
                candidates.append((dist, src_idx, tgt_idx))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        seen: dict[int, int] = {}
        for dist, src, tgt in candidates:
            if seen.get(src, 0) >= maximum_neighbors:
                continue
            delta = (positions[tgt] - positions[src]).abs()
            weight = 1.0 / (1.0 + delta)
            _append_edge(sources, targets, attrs, src, tgt, weight, delta)
            seen[src] = seen.get(src, 0) + 1

    if not sources:
        return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    return (
        torch.tensor([sources, targets], dtype=torch.long, device=device),
        torch.stack(
            [torch.stack(pair) for pair in attrs], dim=0
        ).to(dtype),
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
        self._short_threshold = 4

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

            # Fast path: skip edge building for very short sequences.
            is_short = max(status_count, action_count, entity_count) <= self._short_threshold
            if is_short:
                _d = hierarchy.status.features.device
                _t = hierarchy.status.features.dtype
                for _s, _r, _g in ALL_EDGE_TYPES:
                    self._assign_edges(graph, (_s, _r, _g),
                                       _empty_edge_index(_d), _empty_edge_attr(_d, _t))
                graph.sample_index = torch.tensor(
                    [batch_index], dtype=torch.long, device=_d)
                graphs.append(graph)
                continue

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
