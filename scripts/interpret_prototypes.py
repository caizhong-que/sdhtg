# -*- coding: utf-8 -*-
"""
interpret_prototypes.py -- section 6.6 evidence (3): what do the normal
prototypes actually capture?

Reports, for the learned normal-prototype bank of a finished run:

  * pairwise cosine similarity between prototypes (collapse check);
  * how many samples choose each prototype as nearest (usage), how many
    prototypes are idle, and the entropy of the usage distribution;
  * the soft assignment weights produced by the prototype temperature;
  * representative templates per prototype, i.e. the templates that dominate
    the samples assigned to it (manuscript: representative log snippets).

Usage:
    python scripts/interpret_prototypes.py \
        --root outputs/ssh/main/ladder_full/L7 --seed 42 --split test
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
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


def pairwise_cosine(matrix: torch.Tensor) -> dict[str, float | None]:
    normalized = torch.nn.functional.normalize(matrix.to(torch.float64), dim=-1)
    similarity = normalized @ normalized.T
    count = similarity.shape[0]
    if count < 2:
        return {"mean": None, "min": None, "max": None}
    mask = ~torch.eye(count, dtype=torch.bool)
    values = similarity[mask]
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-templates", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    context = load_context(args.root, args.seed, device=args.device)
    model = context.model
    prototypes = model.detector.normal_prototypes.detach()
    temperature = float(model.config.prototype_temperature)
    num_prototypes = int(prototypes.shape[0])

    data = dataset(context, args.split)
    indices = list(range(min(args.max_samples, len(data))))
    device = default_device(args.device)
    lookup = context.template_lookup()

    usage = [0] * num_prototypes
    soft_totals = [0.0] * num_prototypes
    template_counters = [Counter() for _ in range(num_prototypes)]
    rows: list[dict] = []
    mean_distance = [0.0] * num_prototypes

    for chunk, batch in iter_batches(data, indices, args.batch_size, device):
        result = forward_all(model, batch)
        detector = result["detector"]
        distances = detector.prototype_distances.cpu().to(torch.float64)
        nearest = detector.nearest_prototype.reshape(-1).cpu()
        soft = torch.softmax(-distances / max(temperature, 1e-6), dim=-1)
        probabilities = detector.anomaly_probability.reshape(-1).cpu()
        labels = batch["label"].reshape(-1).cpu()
        template_ids = batch["template_id"].cpu()
        event_mask = batch["mask"].cpu()

        for position, row_index in enumerate(chunk):
            choice = int(nearest[position])
            usage[choice] += 1
            soft_totals[choice] += float(soft[position, choice])
            mean_distance[choice] += float(distances[position, choice])
            length = int(event_mask[position].sum())
            ids = template_ids[position, :length].tolist()
            for value in ids:
                if int(value) > 1:
                    template_counters[choice][int(value)] += 1
            rows.append(
                {
                    "sample_id": str(data.rows[row_index].get("sample_id", row_index)),
                    "label": int(labels[position]),
                    "anomaly_probability": float(probabilities[position]),
                    "nearest_prototype": choice,
                    "nearest_distance": float(distances[position, choice]),
                    "soft_weight": float(soft[position, choice]),
                }
            )

    total = max(len(rows), 1)
    entropy = 0.0
    for count in usage:
        share = count / total
        if share > 0:
            entropy -= share * math.log(share)
    normalised_entropy = entropy / math.log(num_prototypes) if num_prototypes > 1 else 1.0

    prototype_rows = []
    for index in range(num_prototypes):
        assigned = max(usage[index], 1)
        top = template_counters[index].most_common(args.top_templates)
        prototype_rows.append(
            {
                "prototype": index,
                "usage": usage[index],
                "usage_share": usage[index] / total,
                "mean_soft_weight": soft_totals[index] / assigned,
                "mean_nearest_distance": mean_distance[index] / assigned,
                "representative_templates": [
                    {"template_id": key, "count": count, "template": lookup.get(key, f"<{key}>")}
                    for key, count in top
                ],
            }
        )

    summary = {
        "run": str(context.root),
        "seed": args.seed,
        "split": args.split,
        "samples": len(rows),
        "num_prototypes": num_prototypes,
        "prototype_temperature": temperature,
        "prototype_similarity": pairwise_cosine(prototypes),
        "idle_prototypes": [row["prototype"] for row in prototype_rows if row["usage"] == 0],
        "usage_entropy": entropy,
        "usage_entropy_normalised": normalised_entropy,
        "prototypes": prototype_rows,
    }
    out_path = Path(args.out) if args.out else (
        Path(args.root) / "interpretability" / f"prototypes_seed{args.seed}.json"
    )
    write_json(out_path, summary)

    csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "label", "anomaly_probability", "nearest_prototype",
             "nearest_distance", "soft_weight"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["sample_id"],
                    row["label"],
                    f"{row['anomaly_probability']:.6f}",
                    row["nearest_prototype"],
                    f"{row['nearest_distance']:.6f}",
                    f"{row['soft_weight']:.6f}",
                ]
            )

    similarity = summary["prototype_similarity"]
    print(
        f"K={num_prototypes}  samples={len(rows)}  "
        f"cosine(mean/min/max)="
        f"{_fmt(similarity['mean'])}/{_fmt(similarity['min'])}/{_fmt(similarity['max'])}"
    )
    print(
        f"  idle prototypes={summary['idle_prototypes']}  "
        f"usage entropy={normalised_entropy:.3f} (normalised)"
    )
    for row in prototype_rows:
        templates = row["representative_templates"]
        head = templates[0]["template"][:60] if templates else "(unused)"
        print(
            f"  P{row['prototype']:<3} usage={row['usage']:>6} "
            f"({row['usage_share']:.1%})  soft_w={row['mean_soft_weight']:.3f}  "
            f"top template: {head}"
        )
    print(f"\nwrote {out_path}")
    print(f"wrote {csv_path}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
