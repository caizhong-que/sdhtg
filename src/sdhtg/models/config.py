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
    action_prior_scale: float | None = None
    entity_prior_scale: float | None = None
    mode: str = "learned"
    fixed_window_size: int = 16
    random_boundary_prob: float = 0.1
    hard_change_source: str = "action"

    def validate(self) -> None:
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if self.final_temperature <= 0:
            raise ValueError("final_temperature must be positive")
        if self.minimum_temperature <= 0:
            raise ValueError("minimum_temperature must be positive")
        if self.mode not in {"learned", "fixed_window", "random", "hard_change"}:
            raise ValueError(
                f"boundary.mode must be learned/fixed_window/random/hard_change, "
                f"got {self.mode!r}"
            )
        if self.fixed_window_size <= 0:
            raise ValueError("fixed_window_size must be positive")
        if not 0 <= self.random_boundary_prob <= 1:
            raise ValueError("random_boundary_prob must be in [0, 1]")
        if self.hard_change_source not in {"action", "entity"}:
            raise ValueError(
                f"hard_change_source must be action/entity, got {self.hard_change_source!r}"
            )
        for name in ("action_prior_scale", "entity_prior_scale"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")


@dataclass(frozen=True)
class HierarchyConfig:
    membership_epsilon: float = 1e-6
    minimum_node_mass: float = 1e-5
    edge_minimum_weight: float = 1e-6
    max_edges_per_target: int = 10
    normalize_membership: bool = True

    def validate(self) -> None:
        if self.membership_epsilon <= 0:
            raise ValueError("membership_epsilon must be positive")
        if self.minimum_node_mass < 0:
            raise ValueError("minimum_node_mass must be non-negative")
        if self.edge_minimum_weight < 0:
            raise ValueError("edge_minimum_weight must be non-negative")
        if self.max_edges_per_target < 0:
            raise ValueError("max_edges_per_target must be non-negative")


@dataclass(frozen=True)
class AblationConfig:
    use_template_source: bool = True
    use_entity_source: bool = True
    use_action_source: bool = True
    use_status_source: bool = True
    use_time_source: bool = True
    use_hierarchy: bool = True
    use_strategy_film: bool = True
    use_action_boundary: bool = True
    use_entity_boundary: bool = True
    use_temporal_edges: bool = True
    use_semantic_edges: bool = True
    use_cross_level_messages: bool = True
    use_prototypes: bool = True
    detach_boundary_from_graph: bool = False
    single_boundary: bool = False
    independent_boundaries: bool = False
    hard_edge_weight: bool = False


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

    arch: str = "sdhtg"

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
    prototype_similarity_threshold: float = 0.2
    detector_pool_temperature: float = 0.2
    # Manuscript Eq. 54: lambda_p = softplus(eta_p) is learned by default.
    # Setting this to a non-negative value fixes lambda_p instead, which is how
    # the section 6.8 sensitivity sweep varies it.
    prototype_scale_override: float | None = None

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
        if self.prototype_scale_override is not None and (
            self.prototype_scale_override < 0
        ):
            raise ValueError("prototype_scale_override must be non-negative")
        if self.arch not in {"sdhtg", "tcn", "transformer", "gnn_flat",
                             "masked_template"}:
            raise ValueError(
                "arch must be sdhtg/tcn/transformer/gnn_flat/masked_template, "
                f"got {self.arch!r}"
            )
        if self.detector_pool_temperature <= 0:
            raise ValueError("detector_pool_temperature must be positive")
        for level in ("status", "action", "entity"):
            if level not in self.local_temporal_radius:
                raise ValueError(f"missing temporal radius for {level}")
            if level not in self.semantic_neighbors:
                raise ValueError(f"missing semantic neighbor limit for {level}")
        self.boundary.validate()
        self.hierarchy.validate()
