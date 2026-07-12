import torch
import torch.nn.functional as F

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


def _gradient_norm(module) -> float:
    return float(sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    ))


def test_detection_loss_reaches_boundaries_and_encoder():
    torch.manual_seed(5)
    model = SDHTG(tiny_config()).train()
    batch = make_batch(lengths=(8, 7), total_steps=8)
    labels = torch.tensor([0.0, 1.0])

    output = model(batch, boundary_temperature=0.7)
    loss = F.binary_cross_entropy_with_logits(output.anomaly_logit, labels)
    loss.backward()

    assert _gradient_norm(model.detector) > 0
    assert _gradient_norm(model.graph_encoder) > 0
    assert _gradient_norm(model.boundary_network) > 0
    assert _gradient_norm(model.event_encoder) > 0
    assert _gradient_norm(model.strategy_film) > 0


def test_containment_edge_weights_keep_autograd_connection():
    torch.manual_seed(6)
    model = SDHTG(tiny_config()).train()
    output = model(make_batch(lengths=(7,), total_steps=7))
    edge_attr = output.graph_batch[("status", "belongs_to", "action")].edge_attr
    assert edge_attr.requires_grad
    assert edge_attr.grad_fn is not None
