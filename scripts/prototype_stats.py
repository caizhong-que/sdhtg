# -*- coding: utf-8 -*-
"""
prototype_stats.py -- prototype health metrics for one checkpoint.

The manuscript requires the K sweep to report, besides accuracy, the prototype
utilisation, the mean pairwise cosine similarity between prototypes and the
number of idle prototypes (i.e. whether multi-prototype learning collapses).
This script produces exactly those numbers for a single run, so the sweep
runner or the paper table can consume them directly.

Works either from a finished run directory or from a bare checkpoint.

Usage:
    python scripts/prototype_stats.py \
        --root outputs/ssh/main/sens_prototypes_k2 --seed 42
    python scripts/prototype_stats.py \
        --checkpoint outputs/ssh/main/ladder_full/L7/seed_42/checkpoints/best.pt \
        --model-config configs/model/sdhtg.yaml \
        --processed-dir data/processed/ssh
"""

from __future__ import annotations

import argparse
import math
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
from interpret_prototypes import pairwise_cosine
from sdhtg.models.factory import build_model


def load_from_checkpoint(args) -> tuple[torch.nn.Module, Path, torch.device]:
    import json

    processed = Path(args.processed_dir)
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {
        f"{name}_vocab_size": len(vocab[name])
        for name in ("template", "entity", "action", "status")
    }
    device = default_device(args.device)
    model = build_model(args.model_config, overrides).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    return model, processed, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="outputs/<ds>/main/<tag>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.checkpoint:
        if not args.processed_dir:
            raise SystemExit("--checkpoint requires --processed-dir")
        model, processed, device = load_from_checkpoint(args)
        threshold = None
        root = None
    else:
        if not args.root:
            raise SystemExit("pass either --root or --checkpoint")
        context = load_context(args.root, args.seed, device=args.device)
        model, processed, device = context.model, context.processed, context.device
        threshold = context.threshold
        root = context.root

    if not hasattr(model, "detector") or not hasattr(model.detector, "normal_prototypes"):
        raise SystemExit("this checkpoint has no prototype bank (non-SDHTG arch?)")

    prototypes = model.detector.normal_prototypes.detach()
    num_prototypes = int(prototypes.shape[0])
    temperature = float(model.config.prototype_temperature)
    learned_scale = float(
        torch.nn.functional.softplus(model.detector.prototype_scale_logit).item()
    )
    effective_scale = (
        learned_scale
        if model.config.prototype_scale_override is None
        else float(model.config.prototype_scale_override)
    )

    from sdhtg.data.datasets import SessionDataset

    data = SessionDataset(str(Path(processed) / "sessions.parquet"), args.split)
    indices = list(range(min(args.max_samples, len(data))))

    usage = [0] * num_prototypes
    distance_sums = [0.0] * num_prototypes
    for chunk, batch in iter_batches(data, indices, args.batch_size, device):
        result = forward_all(model, batch)
        detector = result["detector"]
        nearest = detector.nearest_prototype.reshape(-1).cpu()
        distances = detector.prototype_distances.cpu()
        for position in range(len(chunk)):
            choice = int(nearest[position])
            usage[choice] += 1
            distance_sums[choice] += float(distances[position, choice])

    total = max(sum(usage), 1)
    entropy = 0.0
    for count in usage:
        share = count / total
        if share > 0:
            entropy -= share * math.log(share)
    normalised_entropy = entropy / math.log(num_prototypes) if num_prototypes > 1 else 1.0

    payload = {
        "run": str(root) if root else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "seed": args.seed,
        "split": args.split,
        "samples": len(indices),
        "threshold": threshold,
        "num_prototypes": num_prototypes,
        "prototype_temperature": temperature,
        "prototype_scale_learned": learned_scale,
        "prototype_scale_effective": effective_scale,
        "prototype_scale_override": model.config.prototype_scale_override,
        "prototype_center": float(model.detector.prototype_center.detach().item()),
        "prototype_similarity": pairwise_cosine(prototypes),
        "usage": usage,
        "usage_share": [count / total for count in usage],
        "idle_prototypes": [i for i, count in enumerate(usage) if count == 0],
        "usage_entropy": entropy,
        "usage_entropy_normalised": normalised_entropy,
        "mean_nearest_distance": [
            (distance_sums[i] / usage[i]) if usage[i] else None
            for i in range(num_prototypes)
        ],
    }

    out_path = Path(args.out) if args.out else (
        (Path(args.root) / "interpretability" / f"prototype_stats_seed{args.seed}.json")
        if args.root
        else Path("outputs/prototype_stats.json")
    )
    write_json(out_path, payload)

    similarity = payload["prototype_similarity"]
    print(
        f"K={num_prototypes}  tau_p={temperature:g}  "
        f"lambda_p={effective_scale:.4f}"
        f"{' (override)' if model.config.prototype_scale_override is not None else ' (learned)'}"
    )
    print(
        f"  cosine mean/min/max = {similarity['mean']}/{similarity['min']}/{similarity['max']}"
        if similarity["mean"] is not None
        else "  cosine: n/a (K=1)"
    )
    print(
        f"  idle prototypes={payload['idle_prototypes']}  "
        f"usage entropy={normalised_entropy:.3f} (normalised)  "
        f"usage={usage}"
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
