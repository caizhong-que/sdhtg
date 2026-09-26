"""Regression tests for the fixed lambda_p switch used by the K sweep."""

from __future__ import annotations

import pytest
import torch

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


def _logit(model: SDHTG, batch) -> float:
    with torch.no_grad():
        return float(model(batch, boundary_temperature=0.5).anomaly_logit[0])


def test_fixed_scale_matches_learned_scale_when_set_to_learned_value():
    torch.manual_seed(3)
    learned = SDHTG(tiny_config()).eval()
    batch = make_batch()
    learned_logit = _logit(learned, batch)

    softplus_value = float(
        torch.nn.functional.softplus(learned.detector.prototype_scale_logit).item()
    )
    torch.manual_seed(3)
    fixed = SDHTG(tiny_config(prototype_scale_override=softplus_value)).eval()
    fixed.load_state_dict(learned.state_dict(), strict=True)

    assert fixed.config.prototype_scale_override == pytest.approx(softplus_value)
    assert _logit(fixed, batch) == pytest.approx(learned_logit, abs=1e-5)


def test_zero_scale_removes_the_prototype_contribution():
    torch.manual_seed(5)
    model = SDHTG(tiny_config(prototype_scale_override=0.0)).eval()
    batch = make_batch()
    output = model(batch, boundary_temperature=0.5)
    with torch.no_grad():
        weighted = (output.level_weights * output.level_logits).sum(-1)
    assert torch.allclose(output.anomaly_logit, weighted, atol=1e-6)


def test_negative_scale_is_rejected():
    with pytest.raises(ValueError):
        tiny_config(prototype_scale_override=-1.0)
