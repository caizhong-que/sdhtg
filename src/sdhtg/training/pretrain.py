from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
import torch
from torch import Tensor, nn

from sdhtg.data.collate import move_batch_to_device
from sdhtg.losses.contrastive import ProjectionHead, supervised_info_nce
from .checkpoint import CheckpointManager
from .curriculum import Curriculum
from .reproducibility import hash_state_dict


logger = logging.getLogger("sdhtg.pretrain")


@dataclass
class PretrainResult:
    best_loss: float
    epochs: int
    checkpoint: str
    protocol: str


class LabelFilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, protocol: str):
        if protocol not in {"normal_only", "all_train"}:
            raise ValueError("protocol must be normal_only or all_train")
        self.dataset = dataset
        self.protocol = protocol
        if protocol == "normal_only":
            labels = dataset.labels
            self.indices = [i for i in range(len(dataset)) if labels[i] == 0]
        else:
            self.indices = list(range(len(dataset)))
        if not self.indices:
            raise ValueError(f"pretraining protocol {protocol!r} selected zero samples")

    def __len__(self):
        return len(self.indices)

    @property
    def lengths(self):
        all_lens = self.dataset.lengths
        return [all_lens[i] for i in self.indices]

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def augment_batch(batch: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value.clone() if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
    mask = result["mask"]
    probability = float(config["token_mask_probability"])
    token_mask = (torch.rand_like(result["delta_t"]) < probability) & mask

    # Random UNK replacement for template/action/status (manuscript 4.10.4).
    for key in ("template_id", "action_id", "status_id"):
        result[key] = result[key].masked_fill(token_mask, 1)

    jitter = torch.randn_like(result["delta_t"]) * float(config["time_jitter_std"])
    result["delta_t"] = (
        result["delta_t"] * (1.0 + jitter)
    ).clamp_min(0.0) * mask.to(result["delta_t"].dtype)
    return result


class ContrastivePretrainer:
    def __init__(
        self,
        model: nn.Module,
        projection: ProjectionHead,
        optimizer: torch.optim.Optimizer,
        loader,
        config: dict[str, Any],
        output_dir: str | Path,
        device: str | torch.device,
        protocol: str,
    ):
        self.model = model
        self.projection = projection
        self.optimizer = optimizer
        self.loader = loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.protocol = protocol
        self.amp = bool(config.get("amp", True) and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.checkpoints = CheckpointManager(self.output_dir)
        self.curriculum = Curriculum(config["curriculum"])
        self.global_step = 0
        self.best_loss = float("inf")
        self.patience_count = 0

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp else nullcontext()
        )

    def _payload_model(self) -> nn.Module:
        # CheckpointManager expects one state_dict. This wrapper ensures model and
        # projection head are restored atomically.
        return nn.ModuleDict({"model": self.model, "projection": self.projection})

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.projection.train()
        state = self.curriculum.at(epoch, int(self.config["pretrain_epochs"]))
        losses = []

        pbar = tqdm(self.loader, desc=f"Epoch {epoch}", unit="batch")
        for batch in pbar:
            batch = move_batch_to_device(batch, self.device)
            strategy = self.config["contrastive"].get("negative_strategy", "hard")
            semantic_keys = None
            if strategy == "semantic":
                # PAD (0) and UNK (1) are shared by every padded row; they
                # must not count as semantic overlap or every candidate would
                # be excluded from the negative set.
                template_sets = [
                    set(int(x) for x in row.tolist() if x > 1)
                    for row in batch["template_id"]
                ]
                entity_sets = [
                    set(int(x) for x in row.tolist() if x > 1)
                    for row in batch["entity_id"]
                ]
                semantic_keys = list(zip(template_sets, entity_sets))
            first = augment_batch(batch, self.config["contrastive"])
            second = augment_batch(batch, self.config["contrastive"])
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                combined = {}
                for k in first:
                    if isinstance(first[k], Tensor):
                        combined[k] = torch.cat([first[k], second[k]], dim=0)
                    else:
                        combined[k] = first[k] + second[k]
                output = self.model(
                    combined,
                    boundary_temperature=state.boundary_temperature,
                    film_strength=state.film_strength,
                    return_graph=False,
                )
                z = self.projection(output.graph_embedding)
                first_z, second_z = z.chunk(2, dim=0)
                # Instance-level positives make both protocols valid. In
                # normal_only mode, passing labels would otherwise remove every
                # normal example from the negative set.
                loss = supervised_info_nce(
                    first_z,
                    second_z,
                    labels=batch["label"] if strategy == "supervised" else None,
                    temperature=float(self.config["contrastive"]["temperature"]),
                    hard_negative_k=int(self.config["contrastive"]["hard_negative_k"]),
                    negative_strategy=strategy,
                    semantic_keys=semantic_keys,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.projection.parameters()),
                float(self.config["grad_clip_norm"]),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            losses.append(float(loss.detach()))

        if not losses:
            raise ValueError("pretraining loader produced zero batches")
        logger.info("epoch %d  loss=%.4f  batches=%d", epoch, float(np.mean(losses)), len(self.loader))
        return float(np.mean(losses))

    def _save(self, name: str, epoch: int, loss: float) -> Path:
        metadata = {
            "stage": "contrastive_pretraining",
            "protocol": self.protocol,
            "epoch_loss": loss,
            "model_sha256": hash_state_dict(self.model),
            "projection_sha256": hash_state_dict(self.projection),
        }
        return self.checkpoints.save(
            name,
            model=self._payload_model(),
            optimizer=self.optimizer,
            scheduler=None,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            best_metric=self.best_loss,
            patience_count=self.patience_count,
            metadata=metadata,
        )

    def fit(self, resume: str | Path | None = None) -> PretrainResult:
        start_epoch = 0
        if resume:
            payload = self.checkpoints.load(
                resume,
                model=self._payload_model(),
                optimizer=self.optimizer,
                scaler=self.scaler,
                map_location=self.device,
            )
            if payload["metadata"].get("protocol") != self.protocol:
                raise ValueError("resume checkpoint protocol differs from requested protocol")
            start_epoch = int(payload["epoch"]) + 1
            self.global_step = int(payload["global_step"])
            self.best_loss = float(payload["best_metric"])
            self.patience_count = int(payload["patience_count"])

        history = []
        maximum = int(self.config["pretrain_epochs"])
        patience = int(self.config.get("pretrain_patience", maximum))
        minimum_delta = float(self.config.get("pretrain_minimum_delta", 0.0))

        for epoch in range(start_epoch, maximum):
            loss = self.train_epoch(epoch)
            improved = loss < self.best_loss - minimum_delta
            if improved:
                self.best_loss = loss
                self.patience_count = 0
                logger.info("  best  loss=%.4f", loss)
            else:
                self.patience_count += 1
                logger.info("  no_improve  loss=%.4f  pat=%d/%d", loss, self.patience_count, patience)

            history.append({"epoch": epoch, "loss": loss})
            self._save("pretrain_last", epoch, loss)
            if improved:
                self._save("pretrain_best", epoch, loss)

            (self.output_dir / "pretrain_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if self.patience_count >= patience:
                break

        best_path = self.output_dir / "pretrain_best.pt"
        if not best_path.exists():
            raise RuntimeError("pretraining did not produce a best checkpoint")
        self.checkpoints.load(
            best_path,
            model=self._payload_model(),
            map_location=self.device,
        )

        # pretrained.pt is an inference/fine-tuning artifact without optimizer
        # state. It contains both encoder and projection state plus provenance.
        destination = self.output_dir / "pretrained.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "projection": self.projection.state_dict(),
                "protocol": self.protocol,
                "best_loss": self.best_loss,
                "global_step": self.global_step,
                "model_sha256": hash_state_dict(self.model),
                "projection_sha256": hash_state_dict(self.projection),
            },
            temporary,
        )
        temporary.replace(destination)
        return PretrainResult(
            best_loss=self.best_loss,
            epochs=len(history),
            checkpoint=str(destination),
            protocol=self.protocol,
        )
