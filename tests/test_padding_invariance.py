import torch

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


def test_padding_does_not_change_prediction():
    torch.manual_seed(3)
    model = SDHTG(tiny_config()).eval()
    compact = make_batch(lengths=(5,), total_steps=5)
    padded = make_batch(lengths=(5,), total_steps=11)

    # make_batch uses the same seeded prefix, but explicitly copy all valid data
    # to make this test independent of random-number consumption order.
    for key in compact:
        if key == "mask":
            continue
        padded[key][:, :5] = compact[key]
    padded["mask"][:] = False
    padded["mask"][:, :5] = True

    with torch.no_grad():
        first = model(compact)
        second = model(padded)

    assert torch.allclose(
        first.anomaly_logit, second.anomaly_logit, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        first.graph_embedding, second.graph_embedding, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        first.event_encoding, second.event_encoding[:, :5], atol=1e-6, rtol=1e-6
    )
