from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from sdhtg.data.collate import collate_sessions
from sdhtg.data.datasets import SessionDataset
from sdhtg.losses.composite import CompositeLoss
from sdhtg.losses.contrastive import ProjectionHead
from sdhtg.models.factory import build_model
from sdhtg.training.pretrain import ContrastivePretrainer, LabelFilteredDataset
from sdhtg.training.reproducibility import seed_everything, write_training_manifest
from sdhtg.training.trainer import Trainer


logger = logging.getLogger("sdhtg")


class BucketBatchSampler(Sampler):
    """Groups indices by quantized length; shorter sequences use larger batches."""
    BUCKET_KEYS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True):
        buckets: dict[int, list[int]] = {}
        for idx, l in enumerate(lengths):
            for b in self.BUCKET_KEYS:
                if l <= b:
                    buckets.setdefault(b, []).append(idx)
                    break

        self.batches: list[list[int]] = []
        for _, indices in sorted(buckets.items()):
            key = _
            # Safe batch sizes: short sequences need tiny [B,S,S] matrices
            if key <= 4:   bs = max(512, batch_size)          # ≤4  events → up to 2048
            elif key <= 8:  bs = min(512, batch_size // 4)    # 5-8  events → 512
            elif key <= 16: bs = min(256, batch_size // 8)    # 9-16 events → 256
            elif key <= 64: bs = 128                          # 17-64 events → 128
            else:           bs = 64                           # >64  events → 64
            bs = min(bs, len(indices))
            rng = np.random.default_rng(seed=(hash(str(indices)) & 0xFFFFFFFF))
            if shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), bs):
                self.batches.append(indices[i:i+bs])
        if shuffle:
            np.random.default_rng(seed=42).shuffle(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def make_loader(dataset, cfg, *, shuffle: bool, seed: int):
    sampler = BucketBatchSampler(dataset.lengths, int(cfg["batch_size"]), shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(cfg["data"]["num_workers"]),
        collate_fn=collate_sessions,
        pin_memory=bool(cfg["data"]["pin_memory"]),
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", help="supervised checkpoint")
    parser.add_argument("--pretrain-resume")
    parser.add_argument(
        "--pretrain-protocol",
        choices=("normal_only", "all_train"),
    )
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--load-pretrained", type=str,
                        help="load pretrained model weights and skip pretraining")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger.info("started  seed=%d  device=%s", int(args.seed if args.seed is not None else cfg["seed"]), device.type)

    seed = int(args.seed if args.seed is not None else cfg["seed"])
    seed_everything(seed, bool(cfg["deterministic"]))

    processed = Path(cfg["data"]["processed_dir"])
    parquet = processed / "sessions.parquet"
    logger.info("data  samples=%s  batch=%d  workers=%d",
                cfg["data"]["processed_dir"], cfg.get("batch_size","?"), cfg["data"]["num_workers"])
    vocabulary = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {
        f"{name}_vocab_size": len(vocabulary[name])
        for name in ("template", "entity", "action", "status")
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model_config"], overrides).to(device)

    if args.load_pretrained:
        ckpt = torch.load(args.load_pretrained, map_location=device, weights_only=False)
        sd = ckpt["model"]
        if any(k.startswith("model.") for k in sd):
            sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(sd)
        logger.info("loaded pretrained model from %s", args.load_pretrained)

    train_dataset = SessionDataset(str(parquet), "train")
    validation_dataset = SessionDataset(str(parquet), "validation")
    logger.info("model  params=%d  train=%d  valid=%d", sum(p.numel() for p in model.parameters()), len(train_dataset), len(validation_dataset))
    output = Path(cfg["output_dir"]) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    write_training_manifest(
        output / "training_manifest.json",
        seed=seed,
        config_paths=[args.config, cfg["model_config"]],
        data_paths=[str(parquet), str(processed / "manifest.json")],
        model=model,
    )
    pretrain_result = None
    if not args.skip_pretrain and not args.load_pretrained and int(cfg.get("pretrain_epochs", 0)) > 0:
        protocol = args.pretrain_protocol or cfg.get("pretrain_protocol", "normal_only")
        logger.info("pretrain  protocol=%s  epochs=%d  lr=%g", protocol, cfg["pretrain_epochs"], cfg["pretrain_learning_rate"])
        pretrain_dataset = LabelFilteredDataset(train_dataset, protocol)
        logger.info("pretrain  samples=%d", len(pretrain_dataset))
        pretrain_loader = make_loader(pretrain_dataset, cfg, shuffle=True, seed=seed)
        projection = ProjectionHead(
            model.config.hidden_dim,
            int(cfg["contrastive"]["projection_dim"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(projection.parameters()),
            lr=float(cfg["pretrain_learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        pretrainer = ContrastivePretrainer(
            model=model,
            projection=projection,
            optimizer=optimizer,
            loader=pretrain_loader,
            config=cfg,
            output_dir=output / "pretraining",
            device=device,
            protocol=protocol,
        )
        pretrain_result = pretrainer.fit(args.pretrain_resume)
        logger.info("pretrain  done  best_loss=%.4f  epochs=%d", pretrain_result.best_loss, pretrain_result.epochs)
        (output / "pretrain_result.json").write_text(
            json.dumps(pretrain_result.__dict__, indent=2), encoding="utf-8"
        )

    if args.pretrain_only:
        return

    labels = [int(l) for l in train_dataset.labels]
    counts = torch.tensor([labels.count(0), labels.count(1)])
    logger.info("supervised  train=%d  normal=%d  anomaly=%d  ratio=%.4f", len(labels), labels.count(0), labels.count(1), labels.count(1)/max(len(labels),1))
    if (counts == 0).any():
        raise ValueError("supervised training split must contain both classes")

    train_loader = make_loader(train_dataset, cfg, shuffle=True, seed=seed + 1)
    validation_loader = make_loader(validation_dataset, cfg, shuffle=False, seed=seed)
    criterion = CompositeLoss(cfg["loss"], counts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        train_loader,
        validation_loader,
        cfg,
        output,
        device,
    )
    result = trainer.fit(args.resume)
    logger.info("done  best_%s=%.4f  epochs=%d  threshold=%.4f", cfg["monitor"], result["best_metric"], result["epochs"], result["threshold"]["threshold"])
    result["pretraining"] = pretrain_result.__dict__ if pretrain_result else None
    (output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
