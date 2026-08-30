# -*- coding: utf-8 -*-
"""
evaluate_boundary.py -- detection metrics + Boundary F1 on the synthetic
structure-only dataset.

For every anomalous test sample the true workflow-swap junction positions are
known (boundaries.json). The model's action boundary probabilities pA are
thresholded (excluding the forced first boundary) and compared with the true
junctions within a tolerance window.

Usage:
    python scripts/evaluate_boundary.py \
        --root outputs/bgl_synth/main/ladder_l7 \
        --experiment-config configs/experiment/bgl_synth.yaml \
        --boundaries data/processed/bgl_synth/boundaries.json \
        --seed 42 --tolerance 2
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
)

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--boundaries", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--extraction", choices=("topk", "threshold"), default="topk"
    )
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(Path(args.experiment_config).read_text(encoding="utf-8"))
    processed = Path(config["data"]["processed_dir"])
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {
        f"{name}_vocab_size": len(vocab[name])
        for name in ("template", "entity", "action", "status")
    }
    manifest = json.loads(
        (root / f"seed_{args.seed}" / "training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    model_config = next(
        p for p in manifest["configs"] if "ladder" in p or "model" in p
    )
    model = build_model(model_config, overrides).to(device).eval()
    checkpoint = torch.load(
        root / f"seed_{args.seed}" / "checkpoints" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    state = checkpoint["model"]
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    threshold = json.loads(
        (root / f"seed_{args.seed}" / "threshold.json").read_text(encoding="utf-8")
    )["threshold"]

    dataset = SessionDataset(str(processed / "sessions.parquet"), "test")
    boundaries = json.loads(Path(args.boundaries).read_text(encoding="utf-8"))

    scores, labels, predicted_boundaries = [], [], []
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            batch = move_batch_to_device(
                collate_sessions([row]), device
            )
            output = model(batch)
            scores.append(float(output.anomaly_probability[0].item()))
            labels.append(int(row["label"]))
            if row["label"] == 1 and hasattr(output, "action_boundary"):
                mask = batch["mask"][0]
                probabilities = output.action_boundary[0][mask].float().cpu().numpy()
                true_junctions = [
                    int(x) for x in boundaries.get(row["sample_id"], [])
                ]
                k = len(true_junctions)
                if args.extraction == "topk":
                    positions = np.argsort(probabilities[1:])[::-1][:k]
                    predicted = [int(pos + 1) for pos in positions]
                else:
                    predicted = [
                        int(pos)
                        for pos, value in enumerate(probabilities)
                        if pos > 0 and value >= args.boundary_threshold
                    ]
                predicted_boundaries.append((row["sample_id"], predicted))

    y_true = np.asarray(labels)
    y_score = np.asarray(scores)
    auprc = float(average_precision_score(y_true, y_score))
    prediction = (y_score >= threshold).astype(int)
    f1 = float(f1_score(y_true, prediction, zero_division=0))
    precision = float(precision_score(y_true, prediction, zero_division=0))
    recall = float(recall_score(y_true, prediction, zero_division=0))
    print(f"AUPRC={auprc:.4f}  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}")

    # Boundary F1 over anomalous samples.
    matches, total_predicted, total_true = 0, 0, 0
    for sample_id, predicted in predicted_boundaries:
        true_junctions = [int(x) for x in boundaries.get(sample_id, [])]
        total_true += len(true_junctions)
        total_predicted += len(predicted)
        for true_pos in true_junctions:
            if any(
                abs(true_pos - predicted_pos) <= args.tolerance
                for predicted_pos in predicted
            ):
                matches += 1
    precision_b = matches / max(total_predicted, 1)
    recall_b = matches / max(total_true, 1)
    f1_b = (
        2 * precision_b * recall_b / (precision_b + recall_b)
        if precision_b + recall_b > 0
        else 0.0
    )
    print(
        f"Boundary F1={f1_b:.4f}  (P_b={precision_b:.4f}  R_b={recall_b:.4f}, "
        f"tol=±{args.tolerance}, threshold={args.boundary_threshold})"
    )


if __name__ == "__main__":
    main()
