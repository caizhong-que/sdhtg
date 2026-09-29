# -*- coding: utf-8 -*-
"""train_mlm_baseline.py -- masked-template pretraining baseline (LogBERT route).

Implements the pretraining route of the comparison under the *unified input
protocol* used by every other baseline in the paper: identical event features
(template/entity/action/status/continuous time), identical encoder depth and
width, identical supervised loss (effective-number weighted focal loss), the
same splits, the same threshold calibration and the same early-stopping rule.
The only difference to the flat Transformer baseline is a masked-template
pretraining stage with a weight-tied prediction head.

Protocol the script reproduces from ``scripts/train.py``:

    * bucketed batches, AdamW (lr 1e-4, weight decay 1e-5), grad clip 5, AMP
    * early stopping on validation AUPRC (patience 10), max 25 supervised epochs
    * threshold calibrated on the validation split by maximising F1
    * metrics written to ``<output_dir>/<tag>/seed_<s>/result.json``

Usage:
    python scripts/train_mlm_baseline.py --dataset ssh --seeds 42 --dry-run
    python scripts/train_mlm_baseline.py --dataset ssh --seeds 42 123 256 512 1024
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.evaluation.calibration import calibrate_threshold
from sdhtg.losses.class_balanced import classification_loss
from sdhtg.models.factory import build_model
from sdhtg.training.reproducibility import seed_everything


UNK = 1


class BucketBatchSampler(torch.utils.data.Sampler):
    BUCKET_KEYS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]

    def __init__(self, lengths, batch_size, shuffle=False, seed=42):
        buckets: dict[int, list[int]] = {}
        for index, length in enumerate(lengths):
            for bound in self.BUCKET_KEYS:
                if length <= bound:
                    buckets.setdefault(bound, []).append(index)
                    break
        self.batches: list[list[int]] = []
        rng = np.random.default_rng(seed)
        for key, indices in sorted(buckets.items()):
            if key <= 4:
                size = max(512, batch_size)
            elif key <= 8:
                size = min(512, batch_size // 4)
            elif key <= 16:
                size = min(256, batch_size // 8)
            elif key <= 64:
                size = 128
            else:
                size = 64
            size = min(size, len(indices))
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), size):
                self.batches.append(indices[start:start + size])
        if shuffle:
            rng.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def masked_template_batch(batch, probability: float, generator):
    """Replace a fraction of template ids by UNK and return the targets."""
    template = batch["template_id"]
    mask = batch["mask"] & (template != 0)
    rand = torch.rand(template.shape, device=template.device, generator=generator)
    selected = mask & (rand < probability)
    if selected.sum() == 0:
        return None, None
    targets = template[selected].clone()
    corrupted = template.clone()
    corrupted[selected] = UNK
    batch = dict(batch)
    batch["template_id"] = corrupted
    return batch, targets, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ssh")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--tag", default="mlm_baseline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config or f"configs/experiment/{args.dataset}.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    processed = Path(config["data"]["processed_dir"])
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {f"{name}_vocab_size": len(vocab[name])
                 for name in ("template", "entity", "action", "status")}
    overrides["arch"] = "masked_template"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in args.seeds:
        run_dir = Path(config["output_dir"]) / args.tag / f"seed_{seed}"
        result_path = run_dir / "result.json"
        if result_path.is_file():
            print(f"== {args.dataset} {args.tag} seed {seed}: cached")
            continue
        if args.dry_run:
            print(f"would train {args.dataset} {args.tag} seed {seed} on {device}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {args.dataset} {args.tag} seed {seed}: training on {device}", flush=True)
        seed_everything(seed, False)
        generator = torch.Generator(device=device).manual_seed(seed)

        model = build_model(args.model_config, overrides).to(device)
        train_set = SessionDataset(str(processed / "sessions.parquet"), "train")
        validation_set = SessionDataset(str(processed / "sessions.parquet"), "validation")
        test_set = SessionDataset(str(processed / "sessions.parquet"), "test")
        batch_size = int(config.get("batch_size", 2048))
        train_loader = BucketBatchSampler(train_set.lengths, batch_size, shuffle=True, seed=seed + 1)
        val_loader = BucketBatchSampler(validation_set.lengths, batch_size)
        test_loader = BucketBatchSampler(test_set.lengths, batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        # ---------------------------------------------------- MLM pretraining
        pretrain_losses = []
        model.train()
        for epoch in range(args.pretrain_epochs):
            total_loss = 0.0
            total_tokens = 0
            for batch_indices in train_loader:
                batch = collate_sessions([train_set[i] for i in batch_indices])
                batch = move_batch_to_device(batch, device)
                corrupted, targets, selected = masked_template_batch(
                    batch, args.mask_probability, generator)
                if corrupted is None:
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model.forward_mlm(corrupted)[selected]
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss) * int(targets.numel())
                total_tokens += int(targets.numel())
            mean_loss = total_loss / max(total_tokens, 1)
            pretrain_losses.append(mean_loss)
            print(f"   mlm epoch {epoch}: loss={mean_loss:.4f}", flush=True)

        # ------------------------------------------------------- supervised
        labels = [int(value) for value in train_set.labels]
        counts = torch.tensor([labels.count(0), labels.count(1)], dtype=torch.float32)
        criterion_args = {
            "beta": float(config["loss"].get("effective_number_beta", 0.9999)),
            "gamma": float(config["loss"].get("focal_gamma", 2.0)),
        }

        def evaluate(loader, dataset):
            model.eval()
            scores, truths = [], []
            with torch.inference_mode():
                for batch_indices in loader:
                    batch = collate_sessions([dataset[i] for i in batch_indices])
                    batch = move_batch_to_device(batch, device)
                    output = model(batch)
                    scores.extend(output.anomaly_probability.float().cpu().tolist())
                    truths.extend(batch["label"].cpu().tolist())
            return np.asarray(truths, dtype=int), np.asarray(scores, dtype=float)

        best_metric, best_state, best_epoch, stale = -np.inf, None, 0, 0
        for epoch in range(args.max_epochs):
            model.train()
            for batch_indices in train_loader:
                batch = collate_sessions([train_set[i] for i in batch_indices])
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(batch)
                    loss = classification_loss(
                        output.anomaly_logit, batch["label"].float(), counts,
                        "cb_focal", criterion_args["beta"], criterion_args["gamma"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            truth, score = evaluate(val_loader, validation_set)
            metric = float(average_precision_score(truth, score))
            marker = ""
            if metric > best_metric + 1e-4:
                best_metric, best_epoch, stale = metric, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                marker = " *"
            else:
                stale += 1
            print(f"   epoch {epoch}: val AUPRC={metric:.4f}{marker}", flush=True)
            if stale >= args.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        truth, score = evaluate(val_loader, validation_set)
        threshold = calibrate_threshold(
            truth, score, objective="f1",
            grid_size=int(config.get("threshold", {}).get("grid_size", 2001)))
        truth, score = evaluate(test_loader, test_set)
        predicted = score >= threshold["threshold"]
        result = {
            "best_metric": float(best_metric),
            "threshold": threshold,
            "epochs": int(best_epoch),
            "mlm_epochs": args.pretrain_epochs,
            "mlm_loss": float(pretrain_losses[-1]) if pretrain_losses else None,
            "test": {
                "auprc": float(average_precision_score(truth, score)),
                "auroc": float(roc_auc_score(truth, score)),
                "threshold_applied": float(threshold["threshold"]),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "samples": int(len(truth)),
            },
            "pretraining": {"objective": "masked_template", "epochs": args.pretrain_epochs,
                            "mask_probability": args.mask_probability},
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("   test AUPRC={auprc:.4f} F1={f1:.4f}".format(**result["test"]), flush=True)

    print("masked-template baseline finished"
          if not args.dry_run else "[dry-run] nothing executed")


if __name__ == "__main__":
    main()
