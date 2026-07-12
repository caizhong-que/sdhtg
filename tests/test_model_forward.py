import torch

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


def test_complete_forward_shapes_and_finiteness():
    torch.manual_seed(1)
    model = SDHTG(tiny_config()).eval()
    batch = make_batch(lengths=(7, 4), total_steps=7)
    output = model(batch, boundary_temperature=0.5)

    assert output.anomaly_logit.shape == (2,)
    assert output.anomaly_probability.shape == (2,)
    assert output.level_logits.shape == (2, 3)
    assert output.level_weights.shape == (2, 3)
    assert output.graph_embedding.shape == (2, 32)
    assert output.action_boundary.shape == (2, 7)
    assert output.entity_boundary.shape == (2, 7)
    assert output.status_to_action.shape == (2, 7, 7)
    assert output.action_to_entity.shape == (2, 7, 7)
    assert torch.isfinite(output.anomaly_logit).all()
    assert torch.isfinite(output.graph_embedding).all()
    assert torch.allclose(output.level_weights.sum(-1), torch.ones(2))


def test_single_event_sequence_is_safe():
    model = SDHTG(tiny_config()).eval()
    output = model(make_batch(lengths=(1,), total_steps=1))
    assert output.anomaly_logit.shape == (1,)
    assert torch.isfinite(output.anomaly_logit).all()
    assert output.action_boundary.item() == 1.0
    assert output.entity_boundary.item() == 1.0
