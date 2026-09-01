# -*- coding: utf-8 -*-
"""
evaluate_faithfulness.py -- structure deletion faithfulness (manuscript Eq. 78).

Fid(k) = z(X) - z(X \\ R_k): remove the top-k nodes ranked by node-level
anomaly evidence (level-head logits) and measure the anomaly-logit drop.
Compared against random-k and low-evidence-k deletion; faithful explanations
should produce Fid(top-k) >> Fid(low-k) ~ Fid(random-k).

Node removal is implemented by zeroing the node embeddings after the graph
encoder, before the detector (so deleted nodes contribute no level evidence).

Usage:
    python scripts/evaluate_faithfulness.py \
        --root outputs/bgl_synth/main/ladder_l7_bsup \
        --experiment-config configs/experiment/bgl_synth.yaml \
        --seed 42 --fractions 0.01,0.05,0.10 --max-samples 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.models.graph import GraphEncoderOutput, NODE_TYPES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", default="0.01,0.05,0.10")
    parser.add_argument("--max-samples", type=int, default=80)
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
    fractions = [float(x) for x in args.fractions.split(",")]

    dataset = SessionDataset(str(processed / "sessions.parquet"), "test")
    rng = np.random.default_rng(12345)
    collected: dict[str, dict[float, list[float]]] = {
        "top": {f: [] for f in fractions},
        "low": {f: [] for f in fractions},
        "random": {f: [] for f in fractions},
    }

    with torch.inference_mode():
        for sample_index in range(min(args.max_samples, len(dataset))):
            batch = move_batch_to_device(
                collate_sessions([dataset[sample_index]]), device
            )
            event_output = model.event_encoder(batch)
            strategy_output = model.strategy_film(
                encoded=event_output.encoded,
                delta_t=batch["delta_t"],
                action_change=batch["action_change"],
                entity_change=batch["entity_change"],
                mask=batch["mask"],
                strength=1.0,
                enabled=model.config.ablation.use_strategy_film,
            )
            boundary_output = model.boundary_network(
                encoded=strategy_output.modulated,
                strategy=strategy_output.per_event_strategy,
                delta_t=batch["delta_t"],
                action_change=batch["action_change"],
                entity_change=batch["entity_change"],
                mask=batch["mask"],
                temperature=model.config.boundary.final_temperature,
            )
            hierarchy_output = model.hierarchy(
                event_features=strategy_output.modulated,
                action_boundaries=boundary_output.action_probability,
                entity_boundaries=boundary_output.entity_probability,
                event_mask=batch["mask"],
                action_ids=batch["action_id"],
                entity_ids=batch["entity_id"],
            )
            graph_build = model.graph_builder.build(hierarchy_output)
            graph_output = model.graph_encoder(graph_build.graphs)

            embeddings = {
                node_type: value.clone()
                for node_type, value in graph_output.node_embeddings.items()
            }
            detector_output = model.detector(
                GraphEncoderOutput(
                    graph=graph_output.graph, node_embeddings=embeddings
                ),
                strategy_output.sequence_strategy,
            )
            z0 = float(detector_output.anomaly_logit[0].item())

            # Node-level evidence = level-head logit (higher = more anomalous).
            node_logits: dict[str, torch.Tensor] = {}
            for node_type in NODE_TYPES:
                values = graph_output.node_embeddings[node_type]
                if values.numel() == 0:
                    node_logits[node_type] = values
                    continue
                node_logits[node_type] = model.detector.level_heads[node_type](
                    values
                ).reshape(-1)

            scores, types, global_indices = [], [], []
            for node_type in NODE_TYPES:
                values = graph_output.node_embeddings[node_type]
                logits = node_logits[node_type]
                scores.append(logits.cpu().numpy())
                types.extend([node_type] * values.shape[0])
                global_indices.append(np.arange(values.shape[0]))
            scores = np.concatenate(scores)
            total = scores.shape[0]
            if total == 0:
                continue

            for fraction in fractions:
                k = max(1, int(round(fraction * total)))
                k = min(k, total)
                order = np.argsort(scores)
                top_indices = order[-k:]
                low_indices = order[:k]
                random_indices = rng.choice(total, size=k, replace=False)
                variants = {
                    "top": top_indices,
                    "low": low_indices,
                    "random": random_indices,
                }
                for variant, selected in variants.items():
                    modified = {
                        node_type: value.clone()
                        for node_type, value in graph_output.node_embeddings.items()
                    }
                    # Map global selected indices back to (node_type, local idx).
                    cursor = 0
                    selected_set = set(int(x) for x in selected)
                    for node_type in NODE_TYPES:
                        count = modified[node_type].shape[0]
                        local = [
                            i - cursor
                            for i in selected_set
                            if cursor <= i < cursor + count
                        ]
                        if local:
                            modified[node_type][local] = 0.0
                        cursor += count
                    out = model.detector(
                        GraphEncoderOutput(
                            graph=graph_output.graph,
                            node_embeddings=modified,
                        ),
                        strategy_output.sequence_strategy,
                    )
                    fid = z0 - float(out.anomaly_logit[0].item())
                    collected[variant][fraction].append(fid)

    print(f"Structure-deletion faithfulness  Fid(k) = z(X) - z(X\\R_k)  "
          f"(n={min(args.max_samples, len(dataset))} test samples)")
    print(f"{'fraction':<10}{'Fid(top)':>14}{'Fid(low)':>14}{'Fid(rand)':>14}"
          f"{'ΔFid top-rand':>16}{'ΔFid top-low':>16}")
    for fraction in fractions:
        row = {}
        for variant in ("top", "low", "random"):
            values = np.asarray(collected[variant][fraction])
            row[variant] = (values.mean(), values.std(ddof=1))
        delta_top_random = row["top"][0] - row["random"][0]
        delta_top_low = row["top"][0] - row["low"][0]
        print(
            f"{fraction:<10.2f}"
            f"{row['top'][0]:>8.4f}±{row['top'][1]:<6.4f}"
            f"{row['low'][0]:>8.4f}±{row['low'][1]:<6.4f}"
            f"{row['random'][0]:>8.4f}±{row['random'][1]:<6.4f}"
            f"{delta_top_random:>16.4f}{delta_top_low:>16.4f}"
        )


if __name__ == "__main__":
    main()
