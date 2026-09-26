# -*- coding: utf-8 -*-
"""
interpret_boundaries.py -- section 6.6 evidence (1): do the learned boundaries
land where the structural priors say a segment should end?

For every valid event the model emits an action-boundary probability and an
entity-boundary probability. If the soft boundary really tracks structure,
high probabilities should coincide with action changes, entity changes and
long inter-event gaps. This script reports, per head:

  * AUC of the boundary probability as a predictor of the matching change flag
    (threshold-free), plus AUC against the other change flag and against gaps;
  * P(change | p >= 0.5) versus P(change | p <= 0.1);
  * mean inter-event gap per boundary-probability bin.

Outputs a JSON summary and a CSV of the bins for the figure stage.

Usage:
    python scripts/interpret_boundaries.py \
        --root outputs/hdfs/main/ladder_full/L7 --seed 42 \
        --split test --max-samples 4000 --device cpu
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from _interpret_common import (
    dataset,
    default_device,
    forward_boundary,
    iter_batches,
    load_context,
    write_json,
)


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranks with ties resolved to their average (needed for an exact AUC)."""
    values = values.to(torch.float64)
    order = torch.argsort(values, stable=True)
    ordered = values[order]
    unique, inverse, counts = torch.unique(
        ordered, return_inverse=True, return_counts=True
    )
    starts = torch.cumsum(counts, 0) - counts
    means = starts.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
    rank_ordered = means[inverse]
    ranks = torch.empty_like(rank_ordered)
    ranks[order] = rank_ordered
    return ranks


def rank_auc(scores: list[float], labels: list[float]) -> float | None:
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


def analyse_head(
    probabilities: list[float],
    primary_change: list[float],
    secondary_change: list[float],
    gaps: list[float],
    bins: int,
) -> dict:
    high = [i for i, p in enumerate(probabilities) if p >= 0.5]
    low = [i for i, p in enumerate(probabilities) if p <= 0.1]

    def rate(indices: list[int], values: list[float]) -> float | None:
        if not indices:
            return None
        return sum(values[i] for i in indices) / len(indices)

    bin_edges = [b / bins for b in range(bins + 1)]
    bin_rows = []
    for index in range(bins):
        lower, upper = bin_edges[index], bin_edges[index + 1]
        members = [
            i
            for i, p in enumerate(probabilities)
            if (lower <= p < upper) or (index == bins - 1 and p == upper)
        ]
        bin_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_gap": rate(members, gaps),
                "primary_change_rate": rate(members, primary_change),
                "secondary_change_rate": rate(members, secondary_change),
            }
        )

    change_index = [
        (p, c)
        for p, c in zip(probabilities, primary_change)
        if c in (0.0, 1.0)
    ]

    # Temperature-free check: among the events the model ranks highest, how
    # often does the structural prior actually change? Comparing that with the
    # base rate avoids reading the sharpened sigmoid at tau=0.1 as a decision.
    ordered = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])
    top_count = max(1, len(ordered) // 10)
    top_indices = ordered[:top_count]
    base_rate = rate(range(len(primary_change)), primary_change)
    top_rate = rate(top_indices, primary_change)

    return {
        "events": len(probabilities),
        "mean_probability": (
            sum(probabilities) / len(probabilities) if probabilities else None
        ),
        "primary_change_rate": rate(range(len(primary_change)), primary_change),
        "secondary_change_rate": rate(range(len(secondary_change)), secondary_change),
        "top_decile_primary_change_rate": top_rate,
        "top_decile_lift": (
            None if (top_rate is None or not base_rate) else top_rate / base_rate
        ),
        "auc_primary_change": rank_auc(
            [p for p, _ in change_index], [c for _, c in change_index]
        ),
        "auc_secondary_change": rank_auc(probabilities, secondary_change),
        "auc_gap": rank_auc(probabilities, gaps),
        "high_p": {
            "threshold": 0.5,
            "count": len(high),
            "primary_change_rate": rate(high, primary_change),
        },
        "low_p": {
            "threshold": 0.1,
            "count": len(low),
            "primary_change_rate": rate(low, primary_change),
        },
        "primary_change_rate_when_changed": None,
        "by_bin": bin_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="outputs/<ds>/main/<tag>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    context = load_context(args.root, args.seed, device=args.device)
    data = dataset(context, args.split)
    indices = list(range(min(args.max_samples, len(data))))
    device = default_device(args.device)

    # The first event of every sample always carries a forced boundary
    # (force_first_boundary), so those positions are tracked separately: the
    # interesting question is whether the *learned* boundaries align with the
    # structural priors.
    collected = {
        "action": {"p": [], "primary": [], "secondary": [], "gap": []},
        "entity": {"p": [], "primary": [], "secondary": [], "gap": []},
    }
    learned_only = {
        "action": {"p": [], "primary": [], "secondary": [], "gap": []},
        "entity": {"p": [], "primary": [], "secondary": [], "gap": []},
    }
    for _, batch in iter_batches(data, indices, args.batch_size, device):
        result = forward_boundary(context.model, batch)
        boundary = result["boundary"]
        mask = batch["mask"].cpu()
        gaps = batch["delta_t"].cpu()
        action_change = batch["action_change"].cpu()
        entity_change = batch["entity_change"].cpu()
        probabilities = {
            "action": boundary.action_probability.cpu(),
            "entity": boundary.entity_probability.cpu(),
        }

        for sample in range(mask.shape[0]):
            length = int(mask[sample].sum().item())
            for position in range(length):
                for head, primary, secondary in (
                    ("action", action_change, entity_change),
                    ("entity", entity_change, action_change),
                ):
                    value = float(probabilities[head][sample, position])
                    entry = {
                        "p": value,
                        "primary": float(primary[sample, position]),
                        "secondary": float(secondary[sample, position]),
                        "gap": float(gaps[sample, position]),
                    }
                    collected[head]["p"].append(entry["p"])
                    collected[head]["primary"].append(entry["primary"])
                    collected[head]["secondary"].append(entry["secondary"])
                    collected[head]["gap"].append(entry["gap"])
                    if position > 0:
                        learned_only[head]["p"].append(entry["p"])
                        learned_only[head]["primary"].append(entry["primary"])
                        learned_only[head]["secondary"].append(entry["secondary"])
                        learned_only[head]["gap"].append(entry["gap"])

    summary = {
        "run": str(context.root),
        "seed": args.seed,
        "split": args.split,
        "samples": len(indices),
        "bins": args.bins,
        "action_boundary": analyse_head(
            collected["action"]["p"],
            collected["action"]["primary"],
            collected["action"]["secondary"],
            collected["action"]["gap"],
            args.bins,
        ),
        "entity_boundary": analyse_head(
            collected["entity"]["p"],
            collected["entity"]["primary"],
            collected["entity"]["secondary"],
            collected["entity"]["gap"],
            args.bins,
        ),
        "action_boundary_learned_only": analyse_head(
            learned_only["action"]["p"],
            learned_only["action"]["primary"],
            learned_only["action"]["secondary"],
            learned_only["action"]["gap"],
            args.bins,
        ),
        "entity_boundary_learned_only": analyse_head(
            learned_only["entity"]["p"],
            learned_only["entity"]["primary"],
            learned_only["entity"]["secondary"],
            learned_only["entity"]["gap"],
            args.bins,
        ),
    }

    out_path = Path(args.out) if args.out else (
        Path(args.root) / "interpretability" / f"boundaries_seed{args.seed}.json"
    )
    write_json(out_path, summary)

    csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["head", "scope", "lower", "upper", "count", "mean_gap",
             "primary_change_rate", "secondary_change_rate"]
        )
        for head in ("action", "entity"):
            for scope, key in (
                ("all", f"{head}_boundary"),
                ("learned_only", f"{head}_boundary_learned_only"),
            ):
                for row in summary[key]["by_bin"]:
                    writer.writerow(
                        [
                            head,
                            scope,
                            f"{row['lower']:.2f}",
                            f"{row['upper']:.2f}",
                            row["count"],
                            "" if row["mean_gap"] is None else f"{row['mean_gap']:.4f}",
                            "" if row["primary_change_rate"] is None
                            else f"{row['primary_change_rate']:.4f}",
                            "" if row["secondary_change_rate"] is None
                            else f"{row['secondary_change_rate']:.4f}",
                        ]
                    )

    for head in ("action", "entity"):
        for scope, label in (
            ("all", "all events"),
            ("learned_only", "excluding forced first boundary"),
        ):
            block = summary[f"{head}_boundary_learned_only" if scope == "learned_only"
                            else f"{head}_boundary"]
            high = block["high_p"]
            low = block["low_p"]
            print(
                f"[{head} | {label}] events={block['events']} "
                f"mean_p={_fmt(block['mean_probability'])} "
                f"change_rate={_fmt(block['primary_change_rate'])}"
            )
            print(
                f"    AUC(p -> {head}_change)={_fmt(block['auc_primary_change'])}  "
                f"AUC(p -> other_change)={_fmt(block['auc_secondary_change'])}  "
                f"AUC(p -> gap)={_fmt(block['auc_gap'])}"
            )
            print(
                f"    P(change | p>=0.5)={_fmt(high['primary_change_rate'])} "
                f"(n={high['count']})   "
                f"P(change | p<=0.1)={_fmt(low['primary_change_rate'])} "
                f"(n={low['count']})"
            )
            print(
                f"    top-10% by boundary score: change rate="
                f"{_fmt(block['top_decile_primary_change_rate'])} vs base rate="
                f"{_fmt(block['primary_change_rate'])} "
                f"(lift={_fmt(block['top_decile_lift'])})"
            )
    print(f"\nwrote {out_path}")
    print(f"wrote {csv_path}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
