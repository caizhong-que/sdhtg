# -*- coding: utf-8 -*-
"""evaluate_noise_paired.py -- paired clean-test reference for section 6.5.

The parsing-noise grid (``run_parsing_noise.py``) trains one model per
(kind, protocol, seed) and evaluates it on the *corrupted* test split.  The
clean reference of the paper is a 5-seed mean over the main ladder, i.e. a
different seed set, so a raw difference mixes the noise effect with seed and
training variance.

This script removes that confound: for every finished noise run it reloads the
run's own checkpoint and re-evaluates it on the *clean* test split of
``data/processed/<dataset>``.  Corrupted and clean test scores then come from
the same weights and the same operating point (the threshold stored by the
run), so the difference isolates the effect of the corrupted test data.

The template id space is shared between a cache and its noise variant
(``make_noisy_dataset.py`` rewrites ids inside the source vocabulary and only
*extends* it for ``split``), which is what makes the clean test split readable
by a noise-trained encoder.  The model is therefore built with the noise
cache's vocabulary sizes while the data comes from the clean cache.

Usage:
    python scripts/evaluate_noise_paired.py --dry-run
    python scripts/evaluate_noise_paired.py --kinds replace merge split unk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Sampler

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.training.reproducibility import seed_everything


KINDS = ("replace", "merge", "split", "unk")
PROTOCOLS = ("test_only", "all")


class BucketBatchSampler(Sampler):
    BUCKET_KEYS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]

    def __init__(self, lengths, batch_size, shuffle=False):
        buckets = {}
        for idx, length in enumerate(lengths):
            for bound in self.BUCKET_KEYS:
                if length <= bound:
                    buckets.setdefault(bound, []).append(idx)
                    break
        self.batches = []
        for key, indices in sorted(buckets.items()):
            if key <= 4:
                bsz = max(512, batch_size)
            elif key <= 8:
                bsz = min(512, batch_size // 4)
            elif key <= 16:
                bsz = min(256, batch_size // 8)
            elif key <= 64:
                bsz = 128
            else:
                bsz = 64
            bsz = min(bsz, len(indices))
            for start in range(0, len(indices), bsz):
                self.batches.append(indices[start:start + bsz])

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def evaluate(model, loader, device):
    y_true, y_score = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.inference_mode():
            out = model(batch)
        y_true.extend(batch["label"].cpu().tolist())
        y_score.extend(out.anomaly_probability.float().cpu().tolist())
    return (
        np.asarray(y_true, dtype=int),
        np.asarray(y_score, dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ssh")
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--kinds", nargs="+", default=list(KINDS))
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(f"configs/experiment/{args.dataset}.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    clean_dir = Path(config["data"]["processed_dir"])
    clean_parquet = clean_dir / "sessions.parquet"
    run_root = Path(config["output_dir"]) / f"noise_r{args.rate:g}"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    pending = done = 0
    for kind in args.kinds:
        for protocol in args.protocols:
            noise_dir = clean_dir.with_name(f"{clean_dir.name}_noise_{kind}_{protocol}")
            vocab = json.loads((noise_dir / "vocab.json").read_text(encoding="utf-8"))
            overrides = {
                f"{name}_vocab_size": len(vocab[name])
                for name in ("template", "entity", "action", "status")
            }
            for seed in args.seeds:
                run_dir = run_root / f"{kind}_{protocol}" / f"seed_{seed}"
                checkpoint = run_dir / "checkpoints" / "best.pt"
                target = run_dir / "clean_paired.json"
                if not checkpoint.is_file():
                    print(f"!! missing checkpoint {checkpoint}")
                    continue
                if target.is_file():
                    done += 1
                    continue
                if args.dry_run:
                    pending += 1
                    print(f"== would evaluate {kind}_{protocol} seed {seed}")
                    continue

                seed_everything(seed, False)
                model = build_model(args.model_config, overrides).to(device).eval()
                ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
                state = ckpt["model"]
                if any(key.startswith("model.") for key in state):
                    state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
                model.load_state_dict(state)

                dataset = SessionDataset(str(clean_parquet), "test")
                sampler = BucketBatchSampler(dataset.lengths, args.batch_size)
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=collate_sessions,
                    num_workers=int(config["data"]["num_workers"]),
                    pin_memory=True,
                )
                y_true, y_score = evaluate(model, loader, device)
                threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
                predicted = (y_score >= threshold).astype(int)
                record = {
                    "kind": kind,
                    "protocol": protocol,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_epoch": ckpt.get("epoch"),
                    "clean_test": {
                        "auprc": float(average_precision_score(y_true, y_score)),
                        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
                        "threshold_applied": float(threshold),
                        "precision": float(precision_score(y_true, predicted, zero_division=0)),
                        "recall": float(recall_score(y_true, predicted, zero_division=0)),
                        "f1": float(f1_score(y_true, predicted, zero_division=0)),
                        "samples": int(len(y_true)),
                    },
                }
                target.write_text(json.dumps(record, indent=2), encoding="utf-8")
                print(
                    "== {} {} seed {}: clean AUPRC {:.4f}  F1 {:.4f} (tau={:.4f})".format(
                        kind, protocol, seed,
                        record["clean_test"]["auprc"],
                        record["clean_test"]["f1"],
                        threshold,
                    ),
                    flush=True,
                )
                del model
                torch.cuda.empty_cache()

    if args.dry_run:
        print(f"[dry-run] paired clean evaluation: {pending} pending, {done} cached")
    else:
        print(f"paired clean evaluation finished ({done} runs were already cached)")


if __name__ == "__main__":
    main()
