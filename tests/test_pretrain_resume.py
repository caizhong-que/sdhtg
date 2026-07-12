import torch
from torch.utils.data import DataLoader

from sdhtg.data.collate import collate_sessions
from sdhtg.losses.contrastive import ProjectionHead
from sdhtg.models.sdhtg import SDHTG
from sdhtg.training.pretrain import ContrastivePretrainer
from tests.model_helpers import tiny_config


def row(index):
    return {
        "sample_id": f"s{index}:0", "session_id": f"s{index}", "label": 0,
        "template_ids": [2, 3, 4], "entity_ids": [2, 2, 2],
        "action_ids": [2, 3, 3], "status_ids": [2, 3, 4],
        "delta_t": [0.0, 1.0, 1.0], "action_change": [1.0, 1.0, 0.0],
        "entity_change": [1.0, 0.0, 0.0],
    }


def config(epochs):
    return {
        "amp": False, "pretrain_epochs": epochs, "pretrain_patience": 20,
        "pretrain_minimum_delta": 0.0, "grad_clip_norm": 5.0,
        "curriculum": {
            "warmup_epochs": 1, "boundary_initial_temperature": 1.0,
            "boundary_final_temperature": 0.2, "film_initial_strength": 0.0,
            "film_final_strength": 1.0, "boundary_loss_warmup_epochs": 1,
        },
        "contrastive": {
            "temperature": 0.07, "token_mask_probability": 0.1,
            "time_jitter_std": 0.05, "hard_negative_k": 4,
        },
    }


def build(tmp_path, epochs):
    model = SDHTG(tiny_config())
    projection = ProjectionHead(32, 16)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()), lr=1e-4
    )
    loader = DataLoader([row(i) for i in range(4)], batch_size=4,
                        collate_fn=collate_sessions, shuffle=False)
    trainer = ContrastivePretrainer(model, projection, optimizer, loader,
                                    config(epochs), tmp_path, "cpu", "normal_only")
    return trainer


def test_pretrain_checkpoint_restores_epoch_optimizer_projection_and_rng(tmp_path):
    first = build(tmp_path, 1)
    first.fit()
    checkpoint = tmp_path / "pretrain_last.pt"
    assert checkpoint.exists()
    assert (tmp_path / "pretrained.pt").exists()

    resumed = build(tmp_path, 2)
    result = resumed.fit(checkpoint)
    assert resumed.global_step == 2
    assert result.protocol == "normal_only"
    assert result.epochs == 1
