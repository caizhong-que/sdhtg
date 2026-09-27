# -*- coding: utf-8 -*-
"""
interpret_boundaries.py -- section 6.6 evidence (1): what do the learned
boundary scores encode, and where do they actually cut?

Two questions that must not be conflated:

1. Ranking. Does the boundary score order events the way the structural priors
   would (action change, entity change, long gap)? Reported as AUC and as the
   top-10% lift over the base rate, both computed on the *pre-temperature*
   logits. Ranking on the probabilities is meaningless here: at the inference
   temperature (0.1) a logit of -10 maps to ~5e-44, so every non-forced
   position underflows to 0 and the probabilities tie.

2. Decision. Does the model actually place a boundary? Reported as the mean
   probability, the mean logit and the fraction of positions with logit > 0.
   A model can rank perfectly and still decide "no internal boundary" if all
   logits stay negative.

The forced first boundary is excluded from every statistic, because its change
prior is zero by construction.

Usage:
    python scripts/interpret_boundaries.py \
        --root outputs/hdfs/main/ladder_full/L7 --seed 42 \
        --split test --max-samples 4000 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

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
    _, inverse, counts = torch.unique(
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


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def analyse(head: dict[str, list[float]], bins: int) -> dict:
    """Summarise one boundary head; ``head`` holds the per-event arrays."""
    scores = head["score"]
    learned_scores = head["learned_score"]
    probabilities = head["probability"]
    primary = head["primary"]
    secondary = head["secondary"]
    gaps = head["gap"]
    total = len(scores)
    if total == 0:
        return {"events": 0}

    ordered = sorted(range(total), key=lambda i: -scores[i])
    top_count = max(1, total // 10)
    top_indices = ordered[:top_count]

    def rate(indices: list[int], values: list[float]) -> float | None:
        if not indices:
            return None
        return sum(values[i] for i in indices) / len(indices)

    base_rate = mean_or_none(primary)
    top_rate = rate(top_indices, primary)

    # Equal-count deciles on the score so the histogram stays informative even
    # when the probabilities themselves are all zero.
    deciles = []
    for index in range(bins):
        start = index * total // bins
        end = (index + 1) * total // bins
        members = ordered[start:end]
        deciles.append(
            {
                "decile": index + 1,
                "count": len(members),
                "score_high": scores[members[0]] if members else None,
                "score_low": scores[members[-1]] if members else None,
                "mean_score": mean_or_none([scores[i] for i in members]),
                "mean_probability": mean_or_none([probabilities[i] for i in members]),
                "mean_gap": mean_or_none([gaps[i] for i in members]),
                "primary_change_rate": rate(members, primary),
                "secondary_change_rate": rate(members, secondary),
            }
        )

    positive_logits = [s for s in scores if s > 0.0]
    change_index = [
        (s, c) for s, c in zip(scores, primary) if c in (0.0, 1.0)
    ]
    return {
        "events": total,
        "mean_score": mean_or_none(scores),
        "std_score": (
            math.sqrt(
                sum((s - mean_or_none(scores)) ** 2 for s in scores)
                / max(total - 1, 1)
            )
            if total > 1
            else 0.0
        ),
        "mean_probability": mean_or_none(probabilities),
        "fraction_score_positive": len(positive_logits) / total,
        "auc_score_primary_change": rank_auc(
            [s for s, _ in change_index], [c for _, c in change_index]
        ),
        "auc_learned_score_primary_change": rank_auc(learned_scores, primary),
        "auc_learned_score_gap": rank_auc(learned_scores, gaps),
        "auc_probability_primary_change": rank_auc(probabilities, primary),
        "auc_score_gap": rank_auc(scores, gaps),
        "primary_change_rate": base_rate,
        "top_decile_change_rate": top_rate,
        "top_decile_lift": (
            None if (top_rate is None or not base_rate) else top_rate / base_rate
        ),
        "deciles": deciles,
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
    temperature = float(context.model.config.boundary.final_temperature)
    boundary_config = context.model.config.boundary
    action_prior_scale = (
        boundary_config.prior_logit_scale
        if boundary_config.action_prior_scale is None
        else boundary_config.action_prior_scale
    )
    entity_prior_scale = (
        boundary_config.prior_logit_scale
        if boundary_config.entity_prior_scale is None
        else boundary_config.entity_prior_scale
    )

    collected: dict[str, dict[str, list[float]]] = {
        "action": {
            key: []
            for key in ("score", "learned_score", "probability", "primary",
                        "secondary", "gap")
        },
        "entity": {
            key: []
            for key in ("score", "learned_score", "probability", "primary",
                        "secondary", "gap")
        },
    }

    for _, batch in iter_batches(data, indices, args.batch_size, device):
        result = forward_boundary(context.model, batch)
        boundary = result["boundary"]
        mask = batch["mask"].cpu()
        gaps = batch["delta_t"].cpu()
        action_change = batch["action_change"].cpu()
        entity_change = batch["entity_change"].cpu()
        action_logit = boundary.action_logit.cpu()
        conditional_logit = boundary.entity_conditional_logit.cpu()
        # The prior enters the logit additively, so subtracting the known prior
        # term isolates what the network itself predicts.
        action_learned_logit = action_logit - action_prior_scale * (
            action_change - 0.5
        )
        conditional_learned_logit = conditional_logit - entity_prior_scale * (
            entity_change - 0.5
        )
        action_probability = boundary.action_probability.cpu()
        entity_probability = boundary.entity_probability.cpu()
        # Composed entity score in log space; the product form
        # p_entity = sigmoid(a/tau) * sigmoid(c/tau) makes log p additive.
        entity_log_score = (
            F.logsigmoid(action_logit / temperature)
            + F.logsigmoid(conditional_logit / temperature)
        )
        entity_learned_log_score = (
            F.logsigmoid(action_learned_logit / temperature)
            + F.logsigmoid(conditional_learned_logit / temperature)
        )

        for sample in range(mask.shape[0]):
            length = int(mask[sample].sum().item())
            for position in range(1, length):  # exclude the forced first boundary
                collected["action"]["score"].append(
                    float(action_logit[sample, position])
                )
                collected["action"]["learned_score"].append(
                    float(action_learned_logit[sample, position])
                )
                collected["action"]["probability"].append(
                    float(action_probability[sample, position])
                )
                collected["action"]["primary"].append(
                    float(action_change[sample, position])
                )
                collected["action"]["secondary"].append(
                    float(entity_change[sample, position])
                )
                collected["action"]["gap"].append(float(gaps[sample, position]))

                collected["entity"]["score"].append(
                    float(entity_log_score[sample, position])
                )
                collected["entity"]["learned_score"].append(
                    float(entity_learned_log_score[sample, position])
                )
                collected["entity"]["probability"].append(
                    float(entity_probability[sample, position])
                )
                collected["entity"]["primary"].append(
                    float(entity_change[sample, position])
                )
                collected["entity"]["secondary"].append(
                    float(action_change[sample, position])
                )
                collected["entity"]["gap"].append(float(gaps[sample, position]))

    summary = {
        "run": str(context.root),
        "seed": args.seed,
        "split": args.split,
        "samples": len(indices),
        "bins": args.bins,
        "boundary_temperature": temperature,
        "prior_logit_scale": {
            "action": action_prior_scale,
            "entity": entity_prior_scale,
        },
        "note": (
            "ranking metrics use pre-temperature scores; probability-based AUC "
            "ties when the logits underflow at the inference temperature"
        ),
        "action_boundary": analyse(collected["action"], args.bins),
        "entity_boundary": analyse(collected["entity"], args.bins),
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
            ["head", "decile", "count", "score_high", "score_low", "mean_score",
             "mean_probability", "mean_gap", "primary_change_rate",
             "secondary_change_rate"]
        )
        for head_name in ("action", "entity"):
            for row in summary[f"{head_name}_boundary"]["deciles"]:
                writer.writerow(
                    [
                        head_name,
                        row["decile"],
                        row["count"],
                        _num(row["score_high"]),
                        _num(row["score_low"]),
                        _num(row["mean_score"]),
                        _num(row["mean_probability"], 3e-3),
                        _num(row["mean_gap"]),
                        _num(row["primary_change_rate"]),
                        _num(row["secondary_change_rate"]),
                    ]
                )

    for head_name in ("action", "entity"):
        block = summary[f"{head_name}_boundary"]
        if not block.get("events"):
            continue
        print(
            f"[{head_name}] events={block['events']} "
            f"score(mean/std)={_num(block['mean_score'])}/{_num(block['std_score'])} "
            f"P(score>0)={block['fraction_score_positive']:.5f} "
            f"mean_probability={_num(block['mean_probability'], 3e-3)}"
        )
        print(
            f"    AUC(score -> primary change)="
            f"{_num(block['auc_score_primary_change'])}   "
            f"AUC(learned component)="
            f"{_num(block['auc_learned_score_primary_change'])}   "
            f"AUC(probability -> primary change)="
            f"{_num(block['auc_probability_primary_change'])}   "
            f"AUC(score -> gap)={_num(block['auc_score_gap'])}"
        )
        print(
            f"    top-10% change rate={_num(block['top_decile_change_rate'])} vs "
            f"base={_num(block['primary_change_rate'])} "
            f"(lift={_num(block['top_decile_lift'])})"
        )
    print(f"\nwrote {out_path}")
    print(f"wrote {csv_path}")


def _num(value: float | None, small: float = 1e-4) -> str:
    if value is None:
        return "n/a"
    if value != 0.0 and abs(value) < small:
        return f"{value:.2e}"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
