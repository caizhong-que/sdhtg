# -*- coding: utf-8 -*-
"""diagnose_faithfulness_sign.py -- decompose z before/after node deletion.

The manuscript defines Fid(k) = z(X) - z(X \\ R_k) with z the anomaly logit, but
the reported values are negative for every deletion variant. This script
decomposes z into its two additive parts

    z = z_cls + scale * (d_proto - center)

and repeats the deletion experiment, so the sign of Fid can be attributed.

Usage:
    python scripts/diagnose_faithfulness_sign.py \
        --root outputs/bgl_synth/main/ladder_l7_bsup \
        --experiment-config configs/experiment/bgl_synth.yaml \
        --seed 42 --fractions 0.01,0.05,0.10 --max-samples 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.models.graph import GraphEncoderOutput, NODE_TYPES


def parts(model, graph_output, strategy_output, embeddings):
    detector = model.detector
    pooled = detector.pool(graph_output, strategy_output) if hasattr(detector, "pool") else None
    output = detector(
        GraphEncoderOutput(graph=graph_output.graph, node_embeddings=embeddings),
        strategy_output.sequence_strategy,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--model-config", default=None,
                        help="explicit model config; defaults to the one in the run manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", default="0.01,0.05,0.10")
    parser.add_argument("--max-samples", type=int, default=60)
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(Path(args.experiment_config).read_text(encoding="utf-8"))
    processed = Path(config["data"]["processed_dir"])
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {f"{name}_vocab_size": len(vocab[name])
                 for name in ("template", "entity", "action", "status")}
    if args.model_config:
        model_config = args.model_config
    else:
        manifest = json.loads((root / f"seed_{args.seed}" / "training_manifest.json")
                              .read_text(encoding="utf-8"))
        model_config = next(p for p in manifest["configs"] if "ladder" in p or "model" in p)
    model = build_model(model_config, overrides).to(device).eval()
    checkpoint = torch.load(root / f"seed_{args.seed}" / "checkpoints" / "best.pt",
                            map_location=device, weights_only=False)
    state = checkpoint["model"]
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    fractions = [float(x) for x in args.fractions.split(",")]
    detector = model.detector
    scale = float(F.softplus(detector.prototype_scale_logit).item())
    center = float(detector.prototype_center.item())

    dataset = SessionDataset(str(processed / "sessions.parquet"), "test")
    rng = np.random.default_rng(12345)
    records = []

    with torch.inference_mode():
        for index in range(min(args.max_samples, len(dataset))):
            row = dataset[index]
            batch = move_batch_to_device(collate_sessions([row]), device)
            event_output = model.event_encoder(batch)
            strategy_output = model.strategy_film(
                encoded=event_output.encoded, delta_t=batch["delta_t"],
                action_change=batch["action_change"], entity_change=batch["entity_change"],
                mask=batch["mask"], strength=1.0,
                enabled=model.config.ablation.use_strategy_film)
            boundary_output = model.boundary_network(
                encoded=strategy_output.modulated, strategy=strategy_output.per_event_strategy,
                delta_t=batch["delta_t"], action_change=batch["action_change"],
                entity_change=batch["entity_change"], mask=batch["mask"],
                temperature=model.config.boundary.final_temperature)
            hierarchy_output = model.hierarchy(
                event_features=strategy_output.modulated,
                action_boundaries=boundary_output.action_probability,
                entity_boundaries=boundary_output.entity_probability,
                event_mask=batch["mask"], action_ids=batch["action_id"],
                entity_ids=batch["entity_id"])
            graph_build = model.graph_builder.build(hierarchy_output)
            graph_output = model.graph_encoder(graph_build.graphs)
            base = parts(model, graph_output, strategy_output,
                         graph_output.node_embeddings)
            z0 = float(base.anomaly_logit[0].item())
            z_cls0 = float((base.level_weights[0] * base.level_logits[0]).sum().item())
            proto0 = float(scale * (base.prototype_distance[0].item() - center))

            scores = []
            for node_type in NODE_TYPES:
                values = graph_output.node_embeddings[node_type]
                if values.numel() == 0:
                    continue
                head = detector.level_heads[node_type](values).reshape(-1)
                scores.append(head.cpu().numpy())
            scores = np.concatenate(scores)
            total = scores.shape[0]
            order = np.argsort(scores)

            row_record = {"label": int(row["label"]), "z0": z0, "z_cls0": z_cls0,
                          "proto0": proto0, "nodes": int(total)}
            for fraction in fractions:
                k = min(total, max(1, int(round(fraction * total))))
                variants = {"top": order[-k:], "low": order[:k],
                            "random": rng.choice(total, size=k, replace=False)}
                for name, selected in variants.items():
                    modified = {t: v.clone() for t, v in graph_output.node_embeddings.items()}
                    cursor = 0
                    chosen = set(int(x) for x in selected)
                    for node_type in NODE_TYPES:
                        count = modified[node_type].shape[0]
                        local = [i - cursor for i in chosen if cursor <= i < cursor + count]
                        if local:
                            modified[node_type][local] = 0.0
                        cursor += count
                    out = parts(model, graph_output, strategy_output, modified)
                    z = float(out.anomaly_logit[0].item())
                    z_cls = float((out.level_weights[0] * out.level_logits[0]).sum().item())
                    proto = float(scale * (out.prototype_distance[0].item() - center))
                    row_record[f"{name}_{fraction}_dz"] = z - z0
                    row_record[f"{name}_{fraction}_dzcls"] = z_cls - z_cls0
                    row_record[f"{name}_{fraction}_dproto"] = proto - proto0
            records.append(row_record)

    labels = np.asarray([r["label"] for r in records])
    print(f"samples: {len(records)}  (positives {int(labels.sum())}, "
          f"negatives {int((labels == 0).sum())})")
    print(f"mean z(X) = {np.mean([r['z0'] for r in records]):+.3f}   "
          f"mean z_cls = {np.mean([r['z_cls0'] for r in records]):+.3f}   "
          f"mean d_proto = {np.mean([r['proto0'] for r in records]):.3f}")
    print(f"prototype scale = {scale:.3f} (softplus), center = {center:.3f}\n")
    print(f"{'ratio':>6} {'variant':>8} {'d_z':>9} {'d_z_cls':>9} {'d_proto':>9}")
    for fraction in fractions:
        for name in ("top", "low", "random"):
            dz = np.mean([r[f"{name}_{fraction}_dz"] for r in records])
            dzc = np.mean([r[f"{name}_{fraction}_dzcls"] for r in records])
            dp = np.mean([r[f"{name}_{fraction}_dproto"] for r in records])
            print(f"{fraction:>6.2f} {name:>8} {dz:>+9.3f} {dzc:>+9.3f} {dp:>+9.3f}")
        print("")
    out_path = Path("outputs/faithfulness_sign_diagnosis.json")
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
