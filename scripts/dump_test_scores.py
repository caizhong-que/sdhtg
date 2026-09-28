# -*- coding: utf-8 -*-
"""dump_test_scores.py -- per-sample test scores for PR/ROC curve figures.

The training loop only stores aggregate metrics, so curve figures need a fresh
inference pass over the test split.  This script reuses the run checkpoint and
writes ``<run>/test_scores.csv`` with one row per test sample (label, score,
length group), plus the calibrated threshold from the run.

Usage:
    python scripts/dump_test_scores.py --dataset ssh --level L7 --seeds 42 123
    python scripts/dump_test_scores.py --dataset hdfs --level L7 --seeds 42 \
        --max-samples 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.training.reproducibility import seed_everything


class BucketBatchSampler(torch.utils.data.Sampler):
    BUCKET_KEYS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]

    def __init__(self, lengths, batch_size, indices=None):
        selected = list(range(len(lengths))) if indices is None else sorted(indices)
        buckets: dict[int, list[int]] = {}
        for idx in selected:
            length = lengths[idx]
            for bound in self.BUCKET_KEYS:
                if length <= bound:
                    buckets.setdefault(bound, []).append(idx)
                    break
        self.batches = []
        for key, idxs in sorted(buckets.items()):
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
            bsz = min(bsz, len(idxs))
            for start in range(0, len(idxs), bsz):
                self.batches.append(idxs[start:start + bsz])

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def stratified_indices(frame: pd.DataFrame, max_samples: int | None, seed: int = 42):
    """Keep every sample when no cap is given, otherwise a stratified subsample."""
    if max_samples is None or len(frame) <= max_samples:
        return np.arange(len(frame))
    labels = frame["label"].to_numpy()
    rng = np.random.default_rng(seed)
    keep = []
    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        take = max(1, int(round(max_samples * len(idx) / len(frame))))
        take = min(take, len(idx))
        keep.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ssh")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ladder", default="ladder_full")
    parser.add_argument("--level", default="L7")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    config_path = Path(args.config or f"configs/experiment/{args.dataset}.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    processed = Path(args.data_dir or config["data"]["processed_dir"])
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {f"{name}_vocab_size": len(vocab[name])
                 for name in ("template", "entity", "action", "status")}
    frame = pd.read_parquet(processed / "sessions.parquet", columns=["split", "label"])
    test_mask = frame["split"].to_numpy() == "test"
    test_frame = frame[test_mask].reset_index(drop=True)
    indices = stratified_indices(test_frame, args.max_samples)
    lengths_all = [len(x) for x in pd.read_parquet(
        processed / "sessions.parquet", columns=["entity_ids"])["entity_ids"]]
    lengths = [lengths_all[i] for i in np.flatnonzero(test_mask)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SessionDataset(str(processed / "sessions.parquet"), "test")
    sampler = BucketBatchSampler(lengths, args.batch_size, indices)

    for seed in args.seeds:
        run_dir = Path(config["output_dir"]) / args.ladder / args.level / f"seed_{seed}"
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            print(f"!! missing {checkpoint}")
            continue
        seed_everything(seed, False)
        model = build_model(args.model_config, overrides).to(device).eval()
        state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
        if any(key.startswith("model.") for key in state):
            state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
        model.load_state_dict(state)
        labels, scores, order = [], [], []
        for batch_indices in sampler:
            batch = collate_sessions([dataset[int(i)] for i in batch_indices])
            batch = move_batch_to_device(batch, device)
            with torch.inference_mode():
                out = model(batch)
            labels.extend(batch["label"].cpu().tolist())
            scores.extend(out.anomaly_probability.float().cpu().tolist())
            order.extend(int(i) for i in batch_indices)
        table = pd.DataFrame({
            "index": order,
            "label": np.asarray(labels, dtype=int),
            "score": np.asarray(scores, dtype=float),
        }).sort_values("index")
        target = run_dir / "test_scores.csv"
        table.to_csv(target, index=False)
        print(f"== {args.dataset} {args.level} seed {seed}: {len(table)} rows -> {target}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
