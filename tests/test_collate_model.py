import torch

from sdhtg.data.collate import collate_sessions
from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import tiny_config


def row(sample, length, label):
    values = list(range(2, 2 + length))
    return {
        "sample_id": sample,
        "session_id": sample.split(":")[0],
        "label": label,
        "template_ids": values,
        "entity_ids": [2] * length,
        "action_ids": [2, *([3] * (length - 1))],
        "status_ids": values,
        "delta_t": [0.0, *([1.0] * (length - 1))],
        "action_change": [1.0, *([0.0] * (length - 1))],
        "entity_change": [1.0, *([0.0] * (length - 1))],
    }


def test_parquet_style_rows_collate_and_run():
    batch = collate_sessions([row("s1:0", 5, 0), row("s2:0", 3, 1)])
    assert batch["mask"].tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    output = SDHTG(tiny_config()).eval()(batch)
    assert output.anomaly_logit.shape == (2,)
    assert torch.isfinite(output.anomaly_logit).all()
