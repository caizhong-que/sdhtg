from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from .class_balanced import classification_loss
from .prototype import prototype_balance_loss, prototype_margin_loss
from .boundary import boundary_regularization
import torch.nn.functional as F
from ..models.config import AblationConfig


@dataclass
class CompositeLossOutput:
    total: Tensor
    components: dict[str, Tensor]


class CompositeLoss(nn.Module):
    def __init__(self, config: dict[str, Any], class_counts: Tensor, ablation_config: AblationConfig | None = None):
        super().__init__(); self.config=config; self.ablation_config=ablation_config
        self.register_buffer("class_counts", torch.as_tensor(class_counts, dtype=torch.float32))

    def forward(
        self,
        output,
        labels: Tensor,
        contrastive_loss: Tensor | None = None,
        boundary_scale: float = 1.0,
        action_change: Tensor | None = None,
        entity_change: Tensor | None = None,
        template_id: Tensor | None = None,
    ) -> CompositeLossOutput:
        c=self.config
        # Samples with label == -1 are unlabeled (label-scarcity protocol):
        # they contribute to boundary/self-supervised terms but not to the
        # supervised classification/prototype losses.
        sample_weight = (labels >= 0).to(labels.dtype)
        labels_safe = labels.clamp_min(0.0)
        classification=classification_loss(
            output.anomaly_logit, labels_safe, self.class_counts,
            c.get("loss_type", "cb_focal"),
            c.get("effective_number_beta", 0.9999),
            c.get("focal_gamma", 2.0),
            c.get("label_smoothing", 0.0),
            sample_weight)
        ablation=self.ablation_config
        if ablation is not None and not ablation.use_prototypes:
            prototype=output.anomaly_logit.sum()*0.0
        else:
            prototype=prototype_margin_loss(
                output.prototype_distance, labels_safe, c["prototype_margin"],
                sample_weight)
        if (
            ablation is not None
            and ablation.use_prototypes
            and c.get("use_prototype_diversity", False)
        ):
            diversity = (
                c.get("prototype_diversity_weight", 0.1)
                * output.prototype_diversity
                + c.get("prototype_balance_weight", 0.1)
                * prototype_balance_loss(
                    output.prototype_distances,
                    labels_safe,
                    c.get("prototype_temperature", 0.1),
                    sample_weight=sample_weight,
                )
            )
        else:
            diversity = output.anomaly_logit.sum() * 0.0
        ablation=self.ablation_config
        use_hierarchy = ablation is None or ablation.use_hierarchy
        if use_hierarchy:
            boundary=boundary_regularization(output.action_boundary, output.entity_boundary,
                output.status_node_mask, c["action_boundary_rate"], c["entity_boundary_rate"])
            # Only boundary entropy and rate regularize training; nesting is
            # enforced structurally, and the former overlap/separation terms
            # are kept as diagnostics only (manuscript 4.10.3).
            boundary_total=(c["boundary_entropy_weight"]*boundary["entropy"]
                +c["boundary_rate_weight"]*boundary["rate"])
            hierarchy_loss = output.anomaly_logit.sum() * 0.0
            # Semantic-change auxiliary supervision: pull the soft boundaries
            # toward the action/entity change indicators so boundary learning
            # does not collapse under weak supervision alone.
            aux_weight = float(c.get("boundary_aux_weight", 0.0))
            if aux_weight > 0.0 and template_id is not None:
                valid = output.status_node_mask.to(action_change.dtype)
                # Template change is the most informative semantic-change
                # signal on BGL-like data (action semantics are ~96% UNK).
                shifted = torch.cat(
                    (template_id[:, :1], template_id[:, :-1]), dim=1
                )
                template_change = (
                    template_id != shifted
                ).to(output.action_boundary.dtype)
                action_target = template_change * valid
                entity_target = entity_change.to(output.entity_boundary.dtype)
                aux = F.binary_cross_entropy(
                    output.action_boundary, action_target,
                    weight=valid, reduction="sum",
                ) + F.binary_cross_entropy(
                    output.entity_boundary, entity_target,
                    weight=valid, reduction="sum",
                )
                aux = aux / valid.sum().clamp_min(1.0)
                boundary_total = boundary_total + aux_weight * aux
            else:
                aux = output.anomaly_logit.sum() * 0.0
        else:
            boundary_total = output.anomaly_logit.sum() * 0.0
            hierarchy_loss = output.anomaly_logit.sum() * 0.0
            aux = output.anomaly_logit.sum() * 0.0
        contrastive = output.anomaly_logit.sum()*0 if contrastive_loss is None else contrastive_loss
        total=(c["classification_weight"]*classification
            +c["prototype_weight"]*prototype
            +diversity
            +boundary_scale*c["boundary_weight"]*boundary_total
            +boundary_scale*c.get("hierarchy_weight", 0.0)*hierarchy_loss
            +c["contrastive_weight"]*contrastive)
        parts={"classification":classification,"prototype":prototype,"diversity":diversity,
               "boundary":boundary_total,"hierarchy":hierarchy_loss,"boundary_aux":aux,
               "contrastive":contrastive}
        return CompositeLossOutput(total, parts)
