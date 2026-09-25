# -*- coding: utf-8 -*-
"""
diagnose_step_cost.py -- attribute per-step cost to model stages for a real batch.

Motivation: the fixed-window boundary ablation trains ~3x slower per epoch than
the learned-boundary ablations while GPU utilisation stays low, which points at
host-side (Python) work rather than GPU kernels. This script measures, for one
real batch of a dataset:

  * how many nodes each hierarchy level produces (learned vs rule boundaries),
  * how many edges each graph relation materialises,
  * wall-clock time of every forward stage and of backward.

It is also the tool used for the efficiency attribution reported in the paper.

Usage:
    python scripts/diagnose_step_cost.py \
        --config configs/experiment/hdfs.yaml \
        --model-config configs/model/rq2_detach.yaml \
        --checkpoint outputs/hdfs/main/ladder_full/L7/seed_42/checkpoints/best.pt \
        --samples 512 --split train --device cpu --repeat 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.models.graph import GraphEncoderOutput


NODE_TYPES = ("status", "action", "entity")
EDGE_TYPES = (
    ("status", "temporal", "status"),
    ("action", "temporal", "action"),
    ("entity", "temporal", "entity"),
    ("status", "semantic", "status"),
    ("action", "semantic", "action"),
    ("entity", "semantic", "entity"),
    ("status", "belongs_to", "action"),
    ("action", "belongs_to", "entity"),
    ("action", "contains", "status"),
    ("entity", "contains", "action"),
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(label: str, device: torch.device, bucket: dict[str, list[float]]):
    """Context manager measuring one stage (CUDA-synchronised)."""

    class _Ctx:
        def __enter__(self):
            _sync(device)
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            _sync(device)
            bucket.setdefault(label, []).append(
                (time.perf_counter() - self._t0) * 1000.0
            )

    return _Ctx()


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--length-range",
        default=None,
        help="restrict to sessions with LOW,HIGH events (reproduces a real "
             "BucketBatchSampler bucket, e.g. 17,64)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--torch-profile",
        type=int,
        default=0,
        help="profile N forward iterations with torch.profiler and print the "
             "top CUDA kernels by device time",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--grad",
        action="store_true",
        help="also time the real training step (model -> CompositeLoss -> backward)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    processed = Path(config["data"]["processed_dir"])
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides = {
        f"{name}_vocab_size": len(vocab[name])
        for name in ("template", "entity", "action", "status")
    }
    model = build_model(args.model_config, overrides).to(device).eval()
    if args.checkpoint:
        load_checkpoint(model, Path(args.checkpoint), device)

    dataset = SessionDataset(str(processed / "sessions.parquet"), args.split)
    if args.length_range:
        low, high = (int(x) for x in args.length_range.split(","))
        import pandas as pd

        frame = pd.read_parquet(
            processed / "sessions.parquet", columns=["split", "length"]
        )
        mask = (
            (frame["split"] == args.split)
            & (frame["length"] > low)
            & (frame["length"] <= high)
        )
        candidates = [int(i) for i in frame.index[mask]]
        indices = candidates[: args.batch_size]
    else:
        indices = list(range(min(args.samples, len(dataset))))
    batch = move_batch_to_device(
        collate_sessions([dataset[i] for i in indices]), device
    )

    bucket: dict[str, list[float]] = {}
    node_counts: dict[str, float] = {}
    edge_counts: dict[tuple[str, str, str], int] = {}

    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(args.torch_profile):
                model(batch)
        print("\ntop CUDA kernels by device time")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        return

    repeats = args.repeat
    # no_grad (not inference_mode) so tensors stay usable for the optional
    # backward measurement below.
    with torch.no_grad():
        for step in range(repeats + 1):  # first iteration is warm-up only
            record = step > 0

            def stage(label: str):
                return (
                    timed(label, device, bucket)
                    if record
                    else _null_ctx()
                )

            with stage("event_encoder"):
                event_output = model.event_encoder(batch)
            with stage("strategy_film"):
                strategy_output = model.strategy_film(
                    encoded=event_output.encoded,
                    delta_t=batch["delta_t"],
                    action_change=batch["action_change"],
                    entity_change=batch["entity_change"],
                    mask=batch["mask"],
                    strength=1.0,
                    enabled=model.config.ablation.use_strategy_film,
                )
            with stage("boundary_network"):
                boundary_output = model.boundary_network(
                    encoded=strategy_output.modulated,
                    strategy=strategy_output.per_event_strategy,
                    delta_t=batch["delta_t"],
                    action_change=batch["action_change"],
                    entity_change=batch["entity_change"],
                    mask=batch["mask"],
                    temperature=model.config.boundary.final_temperature,
                )
            with stage("hierarchy"):
                hierarchy_output = model.hierarchy(
                    event_features=strategy_output.modulated,
                    action_boundaries=boundary_output.action_probability,
                    entity_boundaries=boundary_output.entity_probability,
                    event_mask=batch["mask"],
                    action_ids=batch["action_id"],
                    entity_ids=batch["entity_id"],
                )
            with stage("graph_build"):
                graph_build = model.graph_builder.build(hierarchy_output)
            with stage("graph_encoder"):
                graph_output = model.graph_encoder(graph_build.graphs)
            with stage("detector"):
                detector_output = model.detector(
                    GraphEncoderOutput(
                        graph=graph_output.graph,
                        node_embeddings=graph_output.node_embeddings,
                    ),
                    strategy_output.sequence_strategy,
                )

            if record:
                for node_type in NODE_TYPES:
                    node_counts[node_type] = float(
                        getattr(hierarchy_output, node_type).mask.sum().item()
                    )
                for edge_type in EDGE_TYPES:
                    store = graph_build.graphs[0][edge_type]
                    edge_counts[edge_type] = int(store.edge_index.size(1))

    if args.grad:
        import pandas as pd

        from sdhtg.losses.composite import CompositeLoss

        labels = pd.read_parquet(
            processed / "sessions.parquet", columns=["split", "label"]
        )
        train_labels = labels.loc[labels["split"] == "train", "label"]
        counts = train_labels.value_counts().sort_index().tolist()
        criterion = CompositeLoss(
            config["loss"], counts, ablation_config=model.config.ablation
        ).to(device)
        model.train()
        for step in range(2):
            if step == 1:
                _sync(device)
                t0 = time.perf_counter()
            out = model(batch)
            loss = criterion(
                out,
                batch["label"],
                boundary_scale=1.0,
                action_change=batch["action_change"],
                entity_change=batch["entity_change"],
                template_id=batch["template_id"],
                boundary_label=batch.get("boundary_label"),
            ).total
            loss.backward()
            model.zero_grad(set_to_none=True)
            if step == 1:
                _sync(device)
                bucket.setdefault("full_train_step", []).append(
                    (time.perf_counter() - t0) * 1000.0
                )

    events = float(batch["mask"].sum().item())
    print(f"model config : {args.model_config}")
    print(f"checkpoint   : {args.checkpoint or '(random init)'}")
    print(f"batch        : {len(indices)} outer samples, {int(events)} events, "
          f"{args.split} split, device={device}")

    print("\nsame-semantic grouping inside semantic_edges()")
    print("  (cost of the per-group dense m x m top-k; sum(m^2) is the proxy)")
    for node_type in NODE_TYPES:
        level = getattr(hierarchy_output, node_type)
        counts = level.mask.sum(dim=1).cpu().tolist()
        ids = level.semantic_id.detach().cpu()
        participating = sum(1 for count in counts if int(count) >= 2)
        multi_node = sum(1 for count in counts if int(count) >= 2)
        print(f"  {node_type:<8} nodes/session={sum(counts)/max(len(counts),1):6.2f}  "
              f"sessions with >=2 nodes={multi_node}/{len(counts)} "
              f"(the rest skip the per-session graph loops)")
        groups = 0
        runs = 0
        large = 0
        max_m = 0
        sum_sq = 0
        for index, count in enumerate(counts):
            count = int(count)
            if count < 2:
                continue
            values = ids[index, :count]
            values = values[values > 1]
            if values.numel() < 2:
                continue
            _, sizes = torch.unique_consecutive(torch.sort(values).values,
                                                return_counts=True)
            runs += int(sizes.numel())
            for size in sizes.tolist():
                if size < 2:
                    continue
                groups += 1
                large += int(size >= 64)
                max_m = max(max_m, int(size))
                sum_sq += int(size) * int(size)
        print(f"  {node_type:<8} runs={runs:>7}  groups(m>=2)={groups:>6}  "
              f"sum(m^2)={sum_sq:>9}  max_group={max_m:>4}  groups>=64={large}")

    print("\nnodes per level (whole batch)")
    for node_type in NODE_TYPES:
        share = node_counts[node_type] / max(events, 1.0)
        print(f"  {node_type:<8} {int(node_counts[node_type]):>9}   "
              f"({share:.3f} per event)")
    print("\nedges per relation (whole batch)")
    total_edges = 0
    for edge_type in EDGE_TYPES:
        count = edge_counts[edge_type]
        total_edges += count
        print(f"  {edge_type[0]:>6} -{edge_type[1]:<13}-> {edge_type[2]:<6} "
              f"{count:>10}")
    print(f"  {'TOTAL':<32} {total_edges:>10}")

    kpt = model.config.hierarchy.max_edges_per_target
    min_weight = model.config.hierarchy.edge_minimum_weight
    print("\nmembership numerics (soft_segment_membership output)")
    for name, tensor in (
        ("status->action", hierarchy_output.status_to_action),
        ("action->entity", hierarchy_output.action_to_entity),
    ):
        finite = torch.isfinite(tensor)
        print(f"  {name:<16} max={tensor[finite].max().item():.3e}  "
              f"inf={int((tensor == float('inf')).sum().item())}  "
              f"nan={int(torch.isnan(tensor).sum().item())}  "
              f"nonzero_frac={(tensor > min_weight).float().mean().item():.4f}")

    print("\ncontainment candidates (cost of containment_edges)")
    for name, membership, src_level, tgt_level in (
        ("status->action", hierarchy_output.status_to_action,
         hierarchy_output.status, hierarchy_output.action),
        ("action->entity", hierarchy_output.action_to_entity,
         hierarchy_output.action, hierarchy_output.entity),
    ):
        src_counts = src_level.mask.sum(dim=1).cpu().tolist()
        tgt_counts = tgt_level.mask.sum(dim=1).cpu().tolist()
        pairs = 0
        nonzero = 0
        topk_sessions = 0
        used_sessions = 0
        for index, (s, t) in enumerate(zip(src_counts, tgt_counts)):
            s = int(s)
            t = int(t)
            if s == 0 or t == 0:
                continue
            used_sessions += 1
            weights = membership[index, :s, :t]
            pairs += s * t
            nonzero += int((weights > min_weight).sum().item())
            if kpt > 0 and kpt < s:
                topk_sessions += 1
        share = 100.0 * nonzero / max(pairs, 1)
        print(f"  {name:<16} candidate pairs={pairs:>8}  above eps={nonzero:>8} "
              f"({share:5.1f}%)  top-K branch in {topk_sessions}/{used_sessions} sessions")

    print("\nstage wall-clock (ms, mean of recorded repeats)")
    for label, values in sorted(bucket.items(), key=lambda kv: -sum(kv[1])):
        mean_ms = sum(values) / len(values)
        print(f"  {label:<18} {mean_ms:>10.1f}")


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
