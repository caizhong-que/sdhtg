from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .boundaries import BoundaryOutput, NestedSoftBoundaryNetwork
from .config import SDHTGModelConfig
from .detector import DetectorOutput, FlatGRUDetector, HierarchicalAnomalyDetector
from .event_encoder import EventEncoderOutput, MultiSourceEventEncoder
from .graph import GraphEncoderOutput, HeterogeneousTemporalGraphEncoder
from .graph_builder_batched import GraphBuildResult, HeterogeneousGraphBuilder
from .hierarchy import DifferentiableHierarchy, HierarchyOutput
from .strategy import CausalStrategyFiLM, StrategyOutput
from .utils import assert_finite, masked_mean, validate_sequence_batch


@dataclass
class SDHTGOutput:
    anomaly_logit: Tensor
    anomaly_probability: Tensor
    level_logits: Tensor
    level_weights: Tensor
    graph_embedding: Tensor
    prototype_distance: Tensor
    nearest_prototype: Tensor
    prototype_diversity: Tensor

    action_boundary: Tensor
    entity_boundary: Tensor
    action_boundary_logit: Tensor
    entity_conditional_logit: Tensor

    event_encoding: Tensor
    modulated_event_encoding: Tensor
    source_gate: Tensor
    event_strategy: Tensor
    sequence_strategy: Tensor

    status_to_action: Tensor
    action_to_entity: Tensor
    status_node_mask: Tensor
    action_node_mask: Tensor
    entity_node_mask: Tensor

    graph_batch: Any

    def as_loss_dict(self) -> dict[str, Tensor]:
        return {
            "logit": self.anomaly_logit,
            "probability": self.anomaly_probability,
            "embedding": self.graph_embedding,
            "prototype_distance": self.prototype_distance,
            "p_action": self.action_boundary,
            "p_entity": self.entity_boundary,
            "status_to_action": self.status_to_action,
            "action_to_entity": self.action_to_entity,
        }


class SDHTG(nn.Module):
    """
    End-to-end differentiable semantic hierarchical temporal graph model.

    Gradient route:
        detector
          -> heterogeneous messages
          -> differentiable containment edge weights
          -> soft memberships
          -> nested boundary probabilities
          -> event encoder / strategy FiLM
    """

    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.event_encoder = MultiSourceEventEncoder(config)
        self.strategy_film = CausalStrategyFiLM(config)
        self.boundary_network = NestedSoftBoundaryNetwork(config)
        self.hierarchy = DifferentiableHierarchy(config)
        self.graph_builder = HeterogeneousGraphBuilder(config)
        self.graph_encoder = HeterogeneousTemporalGraphEncoder(config)
        self.detector = HierarchicalAnomalyDetector(config)
        self.flat_detector = (
            FlatGRUDetector(config)
            if not config.ablation.use_hierarchy
            else None
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        boundary_temperature: float | None = None,
        film_strength: float = 1.0,
        return_graph: bool = True,
    ) -> SDHTGOutput:
        validate_sequence_batch(batch)
        temperature = (
            self.config.boundary.final_temperature
            if boundary_temperature is None
            else boundary_temperature
        )

        event_output: EventEncoderOutput = self.event_encoder(batch)
        strategy_output: StrategyOutput = self.strategy_film(
            encoded=event_output.encoded,
            delta_t=batch["delta_t"],
            action_change=batch["action_change"],
            entity_change=batch["entity_change"],
            mask=batch["mask"],
            strength=film_strength,
            enabled=self.config.ablation.use_strategy_film,
        )

        if not self.config.ablation.use_hierarchy:
            pooled = masked_mean(
                strategy_output.modulated, batch["mask"], dim=1
            )
            detector_output = self.flat_detector(
                pooled, strategy_output.sequence_strategy
            )
            batch_size, steps = batch["mask"].shape
            device = strategy_output.modulated.device
            dtype = strategy_output.modulated.dtype
            zero_boundary = torch.zeros(
                batch_size, steps, device=device, dtype=dtype
            )
            empty_matrix = torch.zeros(
                batch_size, steps, 0, device=device, dtype=dtype
            )
            assert_finite(
                (
                    detector_output.anomaly_logit,
                    detector_output.graph_embedding,
                ),
                context="SDHTG forward (flat GRU)",
            )
            return SDHTGOutput(
                anomaly_logit=detector_output.anomaly_logit,
                anomaly_probability=detector_output.anomaly_probability,
                level_logits=detector_output.level_logits,
                level_weights=detector_output.level_weights,
                graph_embedding=detector_output.graph_embedding,
                prototype_distance=detector_output.prototype_distance,
                nearest_prototype=detector_output.nearest_prototype,
                prototype_diversity=detector_output.prototype_diversity,
                action_boundary=zero_boundary,
                entity_boundary=zero_boundary,
                action_boundary_logit=zero_boundary,
                entity_conditional_logit=zero_boundary,
                event_encoding=event_output.encoded,
                modulated_event_encoding=strategy_output.modulated,
                source_gate=event_output.source_gate,
                event_strategy=strategy_output.per_event_strategy,
                sequence_strategy=strategy_output.sequence_strategy,
                status_to_action=empty_matrix,
                action_to_entity=empty_matrix,
                status_node_mask=batch["mask"],
                action_node_mask=torch.zeros_like(batch["mask"]),
                entity_node_mask=torch.zeros_like(batch["mask"]),
                graph_batch=None,
            )

        boundary_output: BoundaryOutput = self.boundary_network(
            encoded=strategy_output.modulated,
            strategy=strategy_output.per_event_strategy,
            delta_t=batch["delta_t"],
            action_change=batch["action_change"],
            entity_change=batch["entity_change"],
            mask=batch["mask"],
            temperature=temperature,
        )
        hierarchy_output: HierarchyOutput = self.hierarchy(
            event_features=strategy_output.modulated,
            action_boundaries=boundary_output.action_probability,
            entity_boundaries=boundary_output.entity_probability,
            event_mask=batch["mask"],
            action_ids=batch["action_id"],
            entity_ids=batch["entity_id"],
        )
        graph_build: GraphBuildResult = self.graph_builder.build(
            hierarchy_output
        )
        graph_output: GraphEncoderOutput = self.graph_encoder(
            graph_build.graphs
        )
        detector_output: DetectorOutput = self.detector(
            graph_output,
            strategy_output.sequence_strategy,
        )

        assert_finite(
            (
                detector_output.anomaly_logit,
                detector_output.graph_embedding,
                boundary_output.action_probability,
                boundary_output.entity_probability,
            ),
            context="SDHTG forward",
        )

        return SDHTGOutput(
            anomaly_logit=detector_output.anomaly_logit,
            anomaly_probability=detector_output.anomaly_probability,
            level_logits=detector_output.level_logits,
            level_weights=detector_output.level_weights,
            graph_embedding=detector_output.graph_embedding,
            prototype_distance=detector_output.prototype_distance,
            nearest_prototype=detector_output.nearest_prototype,
            prototype_diversity=detector_output.prototype_diversity,
            action_boundary=boundary_output.action_probability,
            entity_boundary=boundary_output.entity_probability,
            action_boundary_logit=boundary_output.action_logit,
            entity_conditional_logit=boundary_output.entity_conditional_logit,
            event_encoding=event_output.encoded,
            modulated_event_encoding=strategy_output.modulated,
            source_gate=event_output.source_gate,
            event_strategy=strategy_output.per_event_strategy,
            sequence_strategy=strategy_output.sequence_strategy,
            status_to_action=hierarchy_output.status_to_action,
            action_to_entity=hierarchy_output.action_to_entity,
            status_node_mask=hierarchy_output.status.mask,
            action_node_mask=hierarchy_output.action.mask,
            entity_node_mask=hierarchy_output.entity.mask,
            graph_batch=graph_output.graph if return_graph else None,
        )

    @torch.no_grad()
    def parameter_report(self) -> dict[str, int]:
        modules = {
            "event_encoder": self.event_encoder,
            "strategy_film": self.strategy_film,
            "boundary_network": self.boundary_network,
            "hierarchy": self.hierarchy,
            "graph_encoder": self.graph_encoder,
            "detector": self.detector,
        }
        if self.flat_detector is not None:
            modules["flat_detector"] = self.flat_detector
        report = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in modules.items()
        }
        report["total"] = sum(parameter.numel() for parameter in self.parameters())
        report["trainable"] = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return report
