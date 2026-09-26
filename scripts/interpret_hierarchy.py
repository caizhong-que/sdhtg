# -*- coding: utf-8 -*-
"""
interpret_hierarchy.py -- section 6.6 evidence (2): which level does the
anomaly evidence come from?

The detector gates the three level heads with a per-sample softmax pi over
{status, action, entity} (manuscript section 4.9). This script reports how pi
is distributed for normal and anomalous outer samples, how often each level
dominates, and whether a level weight is itself discriminative (rank AUC).

Usage:
    python scripts/interpret_hierarchy.py \
        --root outputs/ssh/main/ladder_full/L7 --seed 42 --split test
"""

from __future__ import annotations

import argparse
import csv
import statistics as stats
from pathlib import Path

import torch

from _interpret_common import (
    dataset,
    default_device,
    forward_all,
    iter_batches,
    load_context,
    write_json,
)
from sdhtg.models.graph import NODE_TYPES


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64)
    order = torch.argsort(values, stable=True)
    ordered = values[order]
    _, inverse, counts = torch.unique(
        ordered, return_inverse=True, return_counts=True
    )
    starts = torch.cumsum(counts, 0) - counts
    means = starts.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
    rank_ordered = means[inverse]
    ranks = torch.empty_like(rank_ordered)
    ranks[order] = rank_ordered
    return ranks


def rank_auc(scores: list[float], labels: list[int]) -> float | None:
    if not scores:
        return None
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    label_tensor = torch.tensor(labels, dtype=torch.float64)
    positive = label_tensor > 0.5
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(score_tensor)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    context = load_context(args.root, args.seed, device=args.device)
    data = dataset(context, args.split)
    indices = list(range(min(args.max_samples, len(data))))
    device = default_device(args.device)

    rows: list[dict] = []
    for chunk, batch in iter_batches(data, indices, args.batch_size, device):
        result = forward_all(context.model, batch)
        detector = result["detector"]
        weights = detector.level_weights.cpu()
        logits = detector.level_logits.cpu()
        probabilities = detector.anomaly_probability.reshape(-1).cpu()
        labels = batch["label"].reshape(-1).cpu()
        for position, row_index in enumerate(chunk):
            rows.append(
                {
                    "sample_id": str(data.rows[row_index].get("sample_id", row_index)),
                    "row_index": row_index,
                    "label": int(labels[position]),
                    "anomaly_probability": float(probabilities[position]),
                    "weights": {
                        name: float(weights[position, level])
                        for level, name in enumerate(NODE_TYPES)
                    },
                    "logits": {
                        name: float(logits[position, level])
                        for level, name in enumerate(NODE_TYPES)
                    },
                }
            )

    normal = [row for row in rows if row["label"] == 0]
    anomaly = [row for row in rows if row["label"] == 1]
    labels = [row["label"] for row in rows]

    per_level = {}
    for name in NODE_TYPES:
        normal_weights = [row["weights"][name] for row in normal]
        anomaly_weights = [row["weights"][name] for row in anomaly]
        dominant_all = sum(
            1 for row in rows
            if max(row["weights"], key=row["weights"].get) == name
        )
        dominant_normal = sum(
            1 for row in normal
            if max(row["weights"], key=row["weights"].get) == name
        )
        dominant_anomaly = sum(
            1 for row in anomaly
            if max(row["weights"], key=row["weights"].get) == name
        )
        per_level[name] = {
            "mean_normal": stats.fmean(normal_weights) if normal_weights else None,
            "mean_anomaly": stats.fmean(anomaly_weights) if anomaly_weights else None,
            "std_normal": (
                stats.stdev(normal_weights) if len(normal_weights) > 1 else 0.0
            ),
            "std_anomaly": (
                stats.stdev(anomaly_weights) if len(anomaly_weights) > 1 else 0.0
            ),
            "auc_label": rank_auc([row["weights"][name] for row in rows], labels),
            "dominant_share_all": dominant_all / max(len(rows), 1),
            "dominant_share_normal": dominant_normal / max(len(normal), 1),
            "dominant_share_anomaly": dominant_anomaly / max(len(anomaly), 1),
        }

    summary = {
        "run": str(context.root),
        "seed": args.seed,
        "split": args.split,
        "samples": len(rows),
        "normal": len(normal),
        "anomaly": len(anomaly),
        "levels": list(NODE_TYPES),
        "per_level": per_level,
    }
    out_path = Path(args.out) if args.out else (
        Path(args.root) / "interpretability" / f"hierarchy_seed{args.seed}.json"
    )
    write_json(out_path, summary)

    csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "label", "anomaly_probability"]
            + [f"pi_{name}" for name in NODE_TYPES]
        )
        for row in rows:
            writer.writerow(
                [row["sample_id"], row["label"], f"{row['anomaly_probability']:.6f}"]
                + [f"{row['weights'][name]:.6f}" for name in NODE_TYPES]
            )

    print(f"samples={len(rows)}  normal={len(normal)}  anomaly={len(anomaly)}")
    for name in NODE_TYPES:
        block = per_level[name]
        print(
            f"  pi_{name:<7} normal={_fmt(block['mean_normal'])} "
            f"anomaly={_fmt(block['mean_anomaly'])} "
            f"auc={_fmt(block['auc_label'])} "
            f"dominant={block['dominant_share_all']:.1%}"
        )
    print(f"\nwrote {out_path}")
    print(f"wrote {csv_path}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
