from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from .class_balanced import class_balanced_focal_loss
from .prototype import prototype_margin_loss
from .boundary import boundary_regularization


@dataclass
class CompositeLossOutput:
    total: Tensor
    components: dict[str, Tensor]


class CompositeLoss(nn.Module):
    def __init__(self, config: dict[str, Any], class_counts: Tensor):
        super().__init__(); self.config=config
        self.register_buffer("class_counts", torch.as_tensor(class_counts, dtype=torch.float32))

    def forward(self, output, labels: Tensor, contrastive_loss: Tensor | None = None,
                boundary_scale: float = 1.0) -> CompositeLossOutput:
        c=self.config
        classification=class_balanced_focal_loss(
            output.anomaly_logit, labels, self.class_counts,
            c["effective_number_beta"], c["focal_gamma"], c.get("label_smoothing",0.0))
        prototype=prototype_margin_loss(output.prototype_distance, labels, c["prototype_margin"])
        boundary=boundary_regularization(output.action_boundary, output.entity_boundary,
            output.status_node_mask, c["action_boundary_rate"], c["entity_boundary_rate"])
        contrastive = output.anomaly_logit.sum()*0 if contrastive_loss is None else contrastive_loss
        boundary_total=(c["boundary_entropy_weight"]*boundary["entropy"]
            +c["boundary_rate_weight"]*boundary["rate"]
            +c["boundary_separation_weight"]*boundary["separation"])
        total=(c["classification_weight"]*classification
            +c["prototype_weight"]*prototype
            +boundary_scale*c["boundary_weight"]*boundary_total
            +boundary_scale*c["hierarchy_weight"]*boundary["hierarchy"]
            +c["contrastive_weight"]*contrastive)
        parts={"classification":classification,"prototype":prototype,"boundary":boundary_total,
               "hierarchy":boundary["hierarchy"],"contrastive":contrastive}
        return CompositeLossOutput(total, parts)
