import torch

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


def test_entity_boundary_is_nested_inside_action_boundary():
    torch.manual_seed(2)
    model = SDHTG(tiny_config()).eval()
    batch = make_batch(lengths=(9, 5), total_steps=9)
    output = model(batch, boundary_temperature=0.4)
    assert torch.all(output.entity_boundary <= output.action_boundary + 1e-7)
    assert torch.all(output.action_boundary[~batch["mask"]] == 0)
    assert torch.all(output.entity_boundary[~batch["mask"]] == 0)


def test_membership_rows_sum_to_one_for_valid_events():
    model = SDHTG(tiny_config()).eval()
    batch = make_batch(lengths=(8, 5), total_steps=8)
    output = model(batch)
    row_mass = output.status_to_action.sum(dim=2)
    assert torch.allclose(
        row_mass[batch["mask"]],
        torch.ones_like(row_mass[batch["mask"]]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.all(row_mass[~batch["mask"]] == 0)
