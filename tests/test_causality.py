import torch

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


def test_future_events_do_not_change_prefix_event_or_boundary_outputs():
    torch.manual_seed(4)
    model = SDHTG(tiny_config()).eval()
    original = make_batch(lengths=(10,), total_steps=10)
    changed = {key: value.clone() for key, value in original.items()}

    prefix = 5
    changed["template_id"][:, prefix:] = 47
    changed["entity_id"][:, prefix:] = 11
    changed["action_id"][:, prefix:] = 15
    changed["status_id"][:, prefix:] = 46
    changed["delta_t"][:, prefix:] = 999.0
    changed["action_change"][:, prefix:] = 1.0
    changed["entity_change"][:, prefix:] = 1.0

    with torch.no_grad():
        first = model(original)
        second = model(changed)

    # Graph-level score intentionally uses the complete sequence and may change.
    assert torch.allclose(
        first.event_encoding[:, :prefix],
        second.event_encoding[:, :prefix],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        first.event_strategy[:, :prefix],
        second.event_strategy[:, :prefix],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        first.action_boundary[:, :prefix],
        second.action_boundary[:, :prefix],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        first.entity_boundary[:, :prefix],
        second.entity_boundary[:, :prefix],
        atol=1e-6,
        rtol=1e-6,
    )
