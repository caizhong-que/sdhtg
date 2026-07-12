import pytest
import torch

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_and_backward():
    torch.manual_seed(7)
    model = SDHTG(tiny_config()).cuda().train()
    batch = make_batch(lengths=(8, 5), total_steps=8, device="cuda")
    output = model(batch)
    output.anomaly_logit.sum().backward()
    assert output.anomaly_logit.is_cuda
    assert all(
        parameter.grad is None or parameter.grad.is_cuda
        for parameter in model.parameters()
    )
