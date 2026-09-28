# -*- coding: utf-8 -*-
"""evaluate_seen_unseen.py -- seen/unseen entity group evaluation (section 6.5).

The preprocessing pipeline fits its vocabulary on the training split only, so a
test event whose entity never appeared in training is stored as UNK.  That makes
the "unseen entity" group directly observable in the processed cache: a test
sample belongs to the *unseen* group when at least one of its entity tokens is
UNK, and to the *seen* group otherwise.  We additionally separate samples whose
entity tokens are UNK everywhere ("all-unseen").

The script reuses the checkpoints of the main ladder (no training) and reports,
per group, the size, positive rate, AUPRC/AUROC and precision/recall/F1 at the
threshold that the run calibrated on its validation split.  Groups with fewer
than ``--min-positives`` positives are flagged as statistically uninformative
instead of being silently reported.

Usage:
    python scripts/evaluate_seen_unseen.py --dry-run
    python scripts/evaluate_seen_unseen.py --datasets bgl thunderbird ssh openstack
    python scripts/evaluate_seen_unseen.py --datasets hdfs --seeds 42 --max-seen 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Sampler

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.training.reproducibility import seed_everything


UNK = 1


class BucketBatchSampler(Sampler):
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


def natural_groups(frame: pd.DataFrame) -> np.ndarray:
    """Group name per session, derived from the UNK entity tokens of the cache."""
    names = np.empty(len(frame), dtype=object)
    for row, values in enumerate(frame["entity_ids"]):
        tokens = [int(v) for v in values]
        unseen = sum(1 for v in tokens if v == UNK)
        if not tokens or unseen == 0:
            names[row] = "seen"
        elif unseen == len(tokens):
            names[row] = "all_unseen"
        else:
            names[row] = "partially_unseen"
    return names


def declared_groups(path: str, test_rows: np.ndarray) -> np.ndarray:
    """Group name per test session, read from a holdout manifest CSV."""
    table = pd.read_csv(path).set_index("row")
    missing = [int(r) for r in test_rows if int(r) not in table.index]
    if missing:
        raise SystemExit(f"[fail] groups file misses {len(missing)} test rows")
    return table.loc[[int(r) for r in test_rows], "group"].to_numpy()


def select_indices(names: np.ndarray, max_seen: int | None, seed: int = 42) -> np.ndarray:
    interesting = np.flatnonzero(names != "seen")
    seen = np.flatnonzero(names == "seen")
    if max_seen is not None and len(seen) > max_seen:
        rng = np.random.default_rng(seed)
        seen = np.sort(rng.choice(seen, size=max_seen, replace=False))
    return np.sort(np.concatenate([seen, interesting]))


def group_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, min_positives: int):
    size = int(len(labels))
    positives = int(labels.sum())
    predicted = (scores >= threshold).astype(int)
    record = {
        "samples": size,
        "positives": positives,
        "positive_rate": float(positives / size) if size else None,
        "threshold_applied": float(threshold),
        "precision": float(precision_score(labels, predicted, zero_division=0)) if size else None,
        "recall": float(recall_score(labels, predicted, zero_division=0)) if size else None,
        "f1": float(f1_score(labels, predicted, zero_division=0)) if size else None,
        "auprc": float(average_precision_score(labels, scores))
        if 0 < positives < size else None,
        "auroc": float(roc_auc_score(labels, scores))
        if 0 < positives < size else None,
        "informative": bool(size >= 50 and positives >= min_positives),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["bgl", "openstack", "ssh", "hdfs", "thunderbird"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--level", default="L7")
    parser.add_argument("--ladder", default="ladder_full")
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--threshold-source", choices=["run", "best"], default="run",
                        help="'run' uses the calibrated threshold, 'best' refits F1 on the evaluated subset")
    parser.add_argument("--max-seen", type=int, default=None,
                        help="cap on the seen group; the unseen group is always kept in full")
    parser.add_argument("--min-positives", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--data-dir", default=None,
                        help="override the processed cache (e.g. an entity-holdout variant); "
                             "the offsets of --groups-file refer to that cache")
    parser.add_argument("--groups-file", default=None,
                        help="CSV with columns row,split,group from make_entity_holdout.py")
    parser.add_argument("--dump-scores", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in args.datasets:
        config = yaml.safe_load(
            Path(f"configs/experiment/{dataset}.yaml").read_text(encoding="utf-8"))
        processed = Path(args.data_dir) if args.data_dir else Path(config["data"]["processed_dir"])
        vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
        overrides = {f"{name}_vocab_size": len(vocab[name])
                     for name in ("template", "entity", "action", "status")}
        frame = pd.read_parquet(processed / "sessions.parquet",
                                columns=["split", "entity_ids", "label"])
        test_rows = np.flatnonzero(frame["split"].to_numpy() == "test")
        frame = frame.iloc[test_rows].reset_index(drop=True)
        if args.groups_file:
            names = declared_groups(args.groups_file, test_rows)
        else:
            names = natural_groups(frame)
        counts = {name: int((names == name).sum()) for name in pd.unique(names)}
        print(f"== {dataset} ({processed.name}): test={len(frame)}  groups={counts}")
        if min(counts.values()) < 50:
            print("   small group: metrics for it are reported but flagged as not identifiable")
        if args.dry_run:
            continue

        indices = select_indices(names, args.max_seen)
        lengths = [len(x) for x in frame["entity_ids"]]
        dataset_view = SessionDataset(str(processed / "sessions.parquet"), "test")
        # The bucket sampler reorders batches by sequence length, so the batch
        # indices are tracked explicitly to keep scores aligned with the groups.
        sampler = BucketBatchSampler(lengths, args.batch_size, indices)
        for seed in args.seeds:
            run_dir = Path(config["output_dir"]) / args.ladder / args.level / f"seed_{seed}"
            checkpoint = run_dir / "checkpoints" / "best.pt"
            if not checkpoint.is_file():
                print(f"   !! missing {checkpoint}")
                continue
            seed_everything(seed, False)
            model = build_model(args.model_config, overrides).to(device).eval()
            state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
            if any(key.startswith("model.") for key in state):
                state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
            model.load_state_dict(state)
            threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]

            labels, scores, order = [], [], []
            for batch_indices in sampler:
                batch = collate_sessions([dataset_view[i] for i in batch_indices])
                batch = move_batch_to_device(batch, device)
                with torch.inference_mode():
                    out = model(batch)
                labels.extend(batch["label"].cpu().tolist())
                scores.extend(out.anomaly_probability.float().cpu().tolist())
                order.extend(int(i) for i in batch_indices)
            labels = np.asarray(labels, dtype=int)
            scores = np.asarray(scores, dtype=float)
            order = np.asarray(order, dtype=int)
            sub_names = names[order]

            record = {
                "dataset": dataset,
                "level": args.level,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "selection": {
                    "max_seen": args.max_seen,
                    "kept_samples": int(len(indices)),
                    "kept_per_group": {name: int((sub_names == name).sum())
                                       for name in pd.unique(sub_names)},
                },
                "threshold_source": args.threshold_source,
                "groups": {},
            }
            parts = {"all": np.ones(len(sub_names), dtype=bool)}
            for name in pd.unique(sub_names):
                parts[str(name)] = sub_names == name
            if "all_unseen" in parts or "partially_unseen" in parts:
                parts["unseen"] = (sub_names == "all_unseen") | (sub_names == "partially_unseen")
            for name, mask in parts.items():
                if mask.sum() == 0:
                    record["groups"][name] = None
                    continue
                local_threshold = threshold
                if args.threshold_source == "best":
                    grid = np.unique(np.quantile(scores[mask], np.linspace(0, 1, 2001)))
                    best = (0.0, float(threshold))
                    for candidate in grid:
                        value = f1_score(labels[mask], (scores[mask] >= candidate).astype(int),
                                         zero_division=0)
                        if value > best[0]:
                            best = (float(value), float(candidate))
                    local_threshold = best[1]
                record["groups"][name] = group_metrics(
                    labels[mask], scores[mask], local_threshold, args.min_positives)
            out_dir = Path(config["output_dir"]) / "seen_unseen"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{args.ladder.replace('/', '_')}_{args.level}_seed_{seed}"
            target = out_dir / f"{stem}.json"
            target.write_text(json.dumps(record, indent=2), encoding="utf-8")
            if args.dump_scores:
                pd.DataFrame({
                    "index": indices,
                    "group": sub_flags,
                    "label": labels,
                    "score": scores,
                }).to_csv(out_dir / f"{stem}_scores.csv", index=False)
            seen = record["groups"]["seen"]
            unseen = record["groups"].get("unseen") or record["groups"].get("holdout")
            print("   seed {}: seen n={} pos={} F1={} | unseen n={} pos={} F1={} ({})".format(
                seed,
                seen["samples"] if seen else 0, seen["positives"] if seen else 0,
                f"{seen['f1']:.4f}" if seen and seen["f1"] is not None else "n/a",
                unseen["samples"] if unseen else 0, unseen["positives"] if unseen else 0,
                f"{unseen['f1']:.4f}" if unseen and unseen["f1"] is not None else "n/a",
                "informative" if unseen and unseen["informative"] else "not identifiable",
            ), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
