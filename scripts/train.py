from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from sdhtg.data.collate import collate_sessions
from sdhtg.data.datasets import SessionDataset
from sdhtg.losses.composite import CompositeLoss
from sdhtg.losses.contrastive import ProjectionHead
from sdhtg.models.factory import build_model
from sdhtg.training.pretrain import ContrastivePretrainer, LabelFilteredDataset
from sdhtg.training.reproducibility import seed_everything, write_training_manifest
from sdhtg.training.trainer import Trainer


def make_loader(dataset, cfg, *, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        generator=generator if shuffle else None,
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
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    seed_everything(seed, bool(cfg["deterministic"]))

    processed = Path(cfg["data"]["processed_dir"])
    parquet = processed / "sessions.parquet"
    vocabulary = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {
        f"{name}_vocab_size": len(vocabulary[name])
        for name in ("template", "entity", "action", "status")
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model_config"], overrides).to(device)
    train_dataset = SessionDataset(str(parquet), "train")
    validation_dataset = SessionDataset(str(parquet), "validation")
    output = Path(cfg["output_dir"]) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    write_training_manifest(
        output / "training_manifest.json",
        seed=seed,
        config_paths=[args.config, cfg["model_config"]],
        data_paths=[str(parquet), str(processed / "manifest.json")],
        model=model,
    )

    train_subset = int(len(train_dataset) * 0.1)
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_subset))
    # val_subset = int(len(validation_dataset) * 0.1)
    # validation_dataset = torch.utils.data.Subset(validation_dataset, range(val_subset))
    pretrain_result = None
    if not args.skip_pretrain and int(cfg.get("pretrain_epochs", 0)) > 0:
        protocol = args.pretrain_protocol or cfg.get("pretrain_protocol", "normal_only")
        pretrain_dataset = LabelFilteredDataset(train_dataset, protocol)
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
        (output / "pretrain_result.json").write_text(
            json.dumps(pretrain_result.__dict__, indent=2), encoding="utf-8"
        )

    if args.pretrain_only:
        return

    labels = [int(row["label"]) for row in train_dataset.rows]
    counts = torch.tensor([labels.count(0), labels.count(1)])
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
    result["pretraining"] = pretrain_result.__dict__ if pretrain_result else None
    (output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
