# -*- coding: utf-8 -*-
"""
evaluate_boundary_baselines.py -- Boundary F1 of PRESET segmentation
strategies on the semi-synthetic dataset, as the control arm for "learning
beats presetting" (RQ2 / paper Table 6).

Preset strategies produce a per-position boundary score; predicted boundaries
are extracted with the same top-k protocol as the learned model (k = number of
true junctions), tolerance +/-2.

Usage:
    python scripts/evaluate_boundary_baselines.py \
        --sessions data/processed/bgl_synth/sessions.parquet \
        --boundaries data/processed/bgl_synth/boundaries.json \
        --tolerance 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def boundary_f1(predicted: list[int], true: list[int], tolerance: int) -> float:
    if not predicted and not true:
        return 1.0
    matches = sum(
        1
        for true_pos in true
        if any(abs(true_pos - pred) <= tolerance for pred in predicted)
    )
    precision = matches / max(len(predicted), 1)
    recall = matches / max(len(true), 1)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--boundaries", required=True)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--windows", default="4,8,16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sessions = pd.read_parquet(args.sessions)
    boundaries = json.loads(Path(args.boundaries).read_text(encoding="utf-8"))
    test = sessions[sessions.split == "test"]
    windows = [int(x) for x in args.windows.split(",")]
    rng = np.random.default_rng(args.seed)

    per_sample: dict[str, list[float]] = {
        "fixed_w4": [], "fixed_w8": [], "fixed_w16": [],
        "random": [], "hard_template": [], "hard_action": [],
        "hard_entity": [],
    }

    for row in test.itertuples(index=False):
        true_junctions = [int(x) for x in boundaries.get(row.sample_id, [])]
        k = len(true_junctions)
        if k == 0:
            continue
        length = len(row.template_ids)
        template_ids = np.asarray(row.template_ids)
        action_ids = np.asarray(row.action_ids)
        entity_ids = np.asarray(row.entity_ids)

        def topk(scores: np.ndarray) -> list[int]:
            scores = scores.copy()
            scores[0] = -np.inf  # forced first boundary is not counted
            order = np.argsort(scores)[::-1][:k]
            return [int(pos) for pos in order if pos > 0]

        strategies = {
            "fixed_w4": lambda: (np.arange(length) % 4 == 0).astype(float),
            "fixed_w8": lambda: (np.arange(length) % 8 == 0).astype(float),
            "fixed_w16": lambda: (np.arange(length) % 16 == 0).astype(float),
            "random": lambda: rng.random(length),
            "hard_template": lambda: np.concatenate(
                ([0.0], (template_ids[1:] != template_ids[:-1]).astype(float))
            ),
            "hard_action": lambda: np.concatenate(
                ([0.0], (action_ids[1:] != action_ids[:-1]).astype(float))
            ),
            "hard_entity": lambda: np.concatenate(
                ([0.0], (entity_ids[1:] != entity_ids[:-1]).astype(float))
            ),
        }
        for name, fn in strategies.items():
            predicted = topk(fn())
            per_sample[name].append(
                boundary_f1(predicted, true_junctions, args.tolerance)
            )

    print(f"Boundary F1 on bgl_synth test (tol=±{args.tolerance}, top-k):")
    for name, values in per_sample.items():
        array = np.asarray(values)
        print(
            f"  {name:<16} BF1={array.mean():.4f}±{array.std(ddof=1):.4f}"
        )


if __name__ == "__main__":
    main()
