from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BoundaryConfig:
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    minimum_temperature: float = 0.05
    prior_logit_scale: float = 1.0
    force_first_boundary: bool = True

    def validate(self) -> None:
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if self.final_temperature <= 0:
            raise ValueError("final_temperature must be positive")
        if self.minimum_temperature <= 0:
            raise ValueError("minimum_temperature must be positive")


@dataclass(frozen=True)
class HierarchyConfig:
    membership_epsilon: float = 1e-6
    minimum_node_mass: float = 1e-5
    normalize_membership: bool = True

    def validate(self) -> None:
        if self.membership_epsilon <= 0:
            raise ValueError("membership_epsilon must be positive")
        if self.minimum_node_mass < 0:
            raise ValueError("minimum_node_mass must be non-negative")


@dataclass(frozen=True)
class AblationConfig:
    use_template_source: bool = True
    use_entity_source: bool = True
    use_action_source: bool = True
    use_status_source: bool = True
    use_time_source: bool = True
    use_strategy_film: bool = True
    use_action_boundary: bool = True
    use_entity_boundary: bool = True
    use_learned_containment: bool = True
    use_temporal_edges: bool = True
    use_semantic_edges: bool = True
    use_cross_level_messages: bool = True
    use_prototypes: bool = True
    detach_boundary_from_graph: bool = False
    single_boundary: bool = False


@dataclass(frozen=True)
class GraphRelationConfig:
    temporal: bool = True
    semantic: bool = True
    containment: bool = True
    reverse_containment: bool = True
    self_loops: bool = True


@dataclass(frozen=True)
class SDHTGModelConfig:
    template_vocab_size: int
    entity_vocab_size: int
    action_vocab_size: int
    status_vocab_size: int

    hidden_dim: int = 256
    strategy_dim: int = 64
    time_dim: int = 32
    level_dim: int = 16
    num_gru_layers: int = 2
    graph_layers: int = 2
    graph_heads: int = 4
    dropout: float = 0.1

    num_normal_prototypes: int = 8
    prototype_temperature: float = 0.1
    detector_pool_temperature: float = 0.2

    local_temporal_radius: Mapping[str, int] = field(
        default_factory=lambda: {"status": 8, "action": 4, "entity": 2}
    )
    semantic_neighbors: Mapping[str, int] = field(
        default_factory=lambda: {"status": 4, "action": 4, "entity": 2}
    )

    graph_relations: GraphRelationConfig = field(default_factory=GraphRelationConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SDHTGModelConfig":
        values = dict(values)
        if "model" in values:
            values = dict(values["model"])

        values["boundary"] = BoundaryConfig(**values.get("boundary", {}))
        values["hierarchy"] = HierarchyConfig(**values.get("hierarchy", {}))
        values["ablation"] = AblationConfig(**values.get("ablation", {}))
        values["graph_relations"] = GraphRelationConfig(
            **values.get("graph_relations", {})
        )
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        vocabularies = (
            self.template_vocab_size,
            self.entity_vocab_size,
            self.action_vocab_size,
            self.status_vocab_size,
        )
        if any(value < 2 for value in vocabularies):
            raise ValueError("all vocabulary sizes must include PAD and UNK")
        if self.hidden_dim <= 0 or self.strategy_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.hidden_dim % self.graph_heads != 0:
            raise ValueError("hidden_dim must be divisible by graph_heads")
        if self.num_gru_layers <= 0 or self.graph_layers <= 0:
            raise ValueError("encoder and graph layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_normal_prototypes <= 0:
            raise ValueError("num_normal_prototypes must be positive")
        if self.detector_pool_temperature <= 0:
            raise ValueError("detector_pool_temperature must be positive")
        for level in ("status", "action", "entity"):
            if level not in self.local_temporal_radius:
                raise ValueError(f"missing temporal radius for {level}")
            if level not in self.semantic_neighbors:
                raise ValueError(f"missing semantic neighbor limit for {level}")
        self.boundary.validate()
        self.hierarchy.validate()
