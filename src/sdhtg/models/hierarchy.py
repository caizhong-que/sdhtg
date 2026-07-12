from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import SDHTGModelConfig
from .utils import safe_normalize, zero_padding


@dataclass
class HierarchyLevel:
    features: Tensor
    mask: Tensor
    mass: Tensor
    positions: Tensor
    semantic_id: Tensor
    membership: Tensor


@dataclass
class HierarchyOutput:
    status: HierarchyLevel
    action: HierarchyLevel
    entity: HierarchyLevel
    status_to_action: Tensor
    action_to_entity: Tensor


def soft_segment_membership(
    boundaries: Tensor,
    mask: Tensor,
    epsilon: float,
    max_traceback: int | None = None,
) -> Tensor:
    """
    Returns membership M[b, event, candidate_segment].

    Candidate segment k starts at event k. Event t belongs to candidate k when:
    1. k <= t;
    2. a boundary starts at k;
    3. no boundary in (k, t] terminates that candidate.

    M[t,k] = p_k * product_{j=k+1..t}(1-p_j)

    This fixed-candidate representation is differentiable and requires no
    discrete segment extraction during training.

    Memory optimisation: when *max_traceback* is set (e.g. 256), candidate
    segments more than *max_traceback* positions away from the current event
    are ignored, reducing peak memory from O(BÂ·SÂ²) to O(BÂ·SÂ·W).
    """
    if boundaries.shape != mask.shape:
        raise ValueError("boundaries and mask must have identical shapes")

    batch, steps = boundaries.shape
    dtype = boundaries.dtype
    device = boundaries.device

    p = boundaries.clamp(min=epsilon, max=1.0 - epsilon)
    p = torch.where(mask, p, torch.zeros_like(p))

    log_survival = torch.log1p(-p.clamp(max=1.0 - epsilon))
    cumulative = torch.cumsum(log_survival, dim=1)

    t = torch.arange(steps, device=device).view(1, steps, 1)
    k = torch.arange(steps, device=device).view(1, 1, steps)
    W = steps if max_traceback is None else min(steps, max_traceback)

    offsets = torch.arange(W, device=device).view(1, 1, W)
    t_idx = torch.arange(steps, device=device).view(1, steps, 1)
    k_idx = (t_idx - offsets).clamp(min=0)
    valid_k = k_idx >= 0
    cum_t = cumulative.unsqueeze(-1)
    k_flat = k_idx.expand(batch, -1, -1).reshape(batch, -1)
    cum_k = torch.gather(cumulative, 1, k_flat).reshape(batch, steps, W)
    membership = (
        torch.gather(p, 1, k_flat).reshape(batch, steps, W)
        * torch.exp(cum_t - cum_k)
    )
    valid_mask = mask.unsqueeze(-1) & mask.gather(1, k_flat).reshape(batch, steps, W).bool() & valid_k
    membership = membership * valid_mask.to(dtype)
    del cum_t, cum_k, valid_k

    # The first valid candidate covers all probability not otherwise assigned.
    row_mass = membership.sum(dim=-1, keepdim=True)
    residual = (1.0 - row_mass).clamp_min(0.0)
    first_candidate = torch.zeros((1, 1, W), device=device, dtype=dtype)
    first_candidate[..., 0] = 1.0
    membership = membership + residual * first_candidate
    membership = membership * valid_mask.to(dtype)
    del residual, first_candidate, valid_mask
    return membership


class DifferentiableHierarchy(nn.Module):
    def __init__(self, config: SDHTGModelConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.action_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.entity_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    @staticmethod
    def _weighted_mode_proxy(ids: Tensor, membership: Tensor) -> Tensor:
        """
        Selects the ID of the maximum-membership event for graph edge creation.

        IDs control only discrete candidate semantic edges. Feature aggregation
        and containment edge weights remain differentiable.
        """
        # membership: [B, event, segment]
        event_index = membership.argmax(dim=1)
        return ids.gather(1, event_index)

    @staticmethod
    def _aggregate(
        values: Tensor,
        membership: Tensor,
        epsilon: float,
    ) -> tuple[Tensor, Tensor]:
        mass = membership.sum(dim=1)
        features = torch.bmm(membership.transpose(1, 2), values)
        features = features / mass.unsqueeze(-1).clamp_min(epsilon)
        return features, mass

    @staticmethod
    def _positions(
        membership: Tensor,
        epsilon: float,
    ) -> Tensor:
        steps = membership.shape[1]
        index = torch.arange(
            steps, device=membership.device, dtype=membership.dtype
        ).view(1, steps, 1)
        mass = membership.sum(dim=1)
        return (membership * index).sum(dim=1) / mass.clamp_min(epsilon)

    def forward(
        self,
        event_features: Tensor,
        action_boundaries: Tensor,
        entity_boundaries: Tensor,
        event_mask: Tensor,
        action_ids: Tensor,
        entity_ids: Tensor,
    ) -> HierarchyOutput:
        epsilon = self.config.hierarchy.membership_epsilon
        minimum_mass = self.config.hierarchy.minimum_node_mass

        status_membership = torch.diag_embed(
            event_mask.to(event_features.dtype)
        )
        status_mass = event_mask.to(event_features.dtype)
        status_positions = torch.arange(
            event_features.shape[1],
            device=event_features.device,
            dtype=event_features.dtype,
        ).unsqueeze(0).expand(event_features.shape[0], -1)

        action_membership = soft_segment_membership(
            action_boundaries, event_mask, epsilon
        )
        action_features, action_mass = self._aggregate(
            event_features, action_membership, epsilon
        )
        action_features = self.action_projection(action_features)
        action_mask = action_mass > minimum_mass
        action_features = zero_padding(action_features, action_mask)
        action_positions = self._positions(action_membership, epsilon)
        action_semantic = self._weighted_mode_proxy(
            action_ids, action_membership
        )

        # Action candidate k receives an entity boundary probability obtained
        # from the event-to-action membership at the candidate start.
        action_entity_boundary = torch.bmm(
            action_membership.transpose(1, 2),
            entity_boundaries.unsqueeze(-1),
        ).squeeze(-1)
        action_entity_boundary = (
            action_entity_boundary
            / action_mass.clamp_min(epsilon)
        ).clamp(0.0, 1.0)
        action_entity_boundary = torch.where(
            action_mask,
            action_entity_boundary,
            torch.zeros_like(action_entity_boundary),
        )
        action_entity_boundary[:, 0] = action_mask[:, 0].to(
            action_entity_boundary.dtype
        )

        action_to_entity = soft_segment_membership(
            action_entity_boundary, action_mask, epsilon
        )
        entity_features, entity_mass = self._aggregate(
            action_features, action_to_entity, epsilon
        )
        entity_features = self.entity_projection(entity_features)
        entity_mask = entity_mass > minimum_mass
        entity_features = zero_padding(entity_features, entity_mask)
        entity_positions = self._positions(action_to_entity, epsilon)
        entity_semantic = self._weighted_mode_proxy(
            self._weighted_mode_proxy(entity_ids, action_membership),
            action_to_entity,
        )

        return HierarchyOutput(
            status=HierarchyLevel(
                features=event_features,
                mask=event_mask,
                mass=status_mass,
                positions=status_positions,
                semantic_id=action_ids,
                membership=status_membership,
            ),
            action=HierarchyLevel(
                features=action_features,
                mask=action_mask,
                mass=action_mass,
                positions=action_positions,
                semantic_id=action_semantic,
                membership=action_membership,
            ),
            entity=HierarchyLevel(
                features=entity_features,
                mask=entity_mask,
                mass=entity_mass,
                positions=entity_positions,
                semantic_id=entity_semantic,
                membership=action_to_entity,
            ),
            status_to_action=action_membership,
            action_to_entity=action_to_entity,
        )
