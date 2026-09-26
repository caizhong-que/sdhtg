# -*- coding: utf-8 -*-
"""
interpret_cases.py -- section 6.6 evidence (4): top-k anomalous subgraph and
the three mandatory case studies (correct detection, false positive, false
negative).

Case selection over the test split:
    correct  -> highest-scoring true positive
    fp       -> highest-scoring false positive
    fn       -> lowest-scoring true positive (confidently missed)

For each case the script re-runs the single sample, ranks graph nodes by their
level-head logit, maps the selected nodes back to the events they were built
from, keeps the highest-weight edges between the selected nodes, and writes a
self-contained JSON that the figure stage can render (timeline + subgraph).

Usage:
    python scripts/interpret_cases.py \
        --root outputs/hdfs/main/ladder_full/L7 --seed 42 --split test \
        --top-k 8 --max-scan 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _interpret_common import (
    dataset,
    default_device,
    forward_all,
    iter_batches,
    load_context,
    node_to_events,
    top_templates,
    write_json,
)
from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.models.graph import EDGE_TYPES, NODE_TYPES


def select_cases(
    rows: list[dict],
    requested: list[str],
    threshold: float | None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Pick the case studies at the calibrated operating point.

    A genuine false positive must cross the threshold while being normal, and a
    genuine false negative must stay below it while being anomalous. When the
    scan contains no such sample the most extreme opposite case is returned and
    flagged, so the paper never reports a near-miss as an error case by accident.
    """
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    chosen: dict[str, dict] = {}
    notes: dict[str, str] = {}
    cut = threshold if threshold is not None else 0.5

    def pick(name: str) -> None:
        if name == "correct":
            candidates = [row for row in positives if row["probability"] >= cut]
            if candidates:
                chosen[name] = max(candidates, key=lambda row: row["probability"])
            elif positives:
                chosen[name] = max(positives, key=lambda row: row["probability"])
                notes[name] = "no_threshold_crossing_true_positive"
        elif name == "fp":
            candidates = [row for row in negatives if row["probability"] >= cut]
            if candidates:
                chosen[name] = max(candidates, key=lambda row: row["probability"])
            elif negatives:
                chosen[name] = max(negatives, key=lambda row: row["probability"])
                notes[name] = "no_threshold_crossing_false_positive"
        elif name == "fn":
            candidates = [row for row in positives if row["probability"] < cut]
            if candidates:
                chosen[name] = min(candidates, key=lambda row: row["probability"])
            elif positives:
                chosen[name] = min(positives, key=lambda row: row["probability"])
                notes[name] = "no_threshold_missed_positive"

    for name in requested:
        pick(name)
    return chosen, notes


def subgraph_payload(
    context,
    data,
    row_index: int,
    top_k: int,
    epsilon: float,
    max_events: int,
    min_mass: float,
) -> dict:
    model = context.model
    device = context.device
    batch = move_batch_to_device(collate_sessions([data[row_index]]), device)
    result = forward_all(model, batch)
    hierarchy = result["hierarchy"]
    graph_output = result["graph"]
    detector = result["detector"]
    graph = result["graph_build"].graphs[0]

    scores: dict[str, torch.Tensor] = {}
    valid_indices: dict[str, list[int]] = {}
    masses: dict[str, torch.Tensor] = {}
    positions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for node_type in NODE_TYPES:
            values = graph_output.node_embeddings[node_type]
            level = getattr(hierarchy, node_type)
            # Active segments can start anywhere, so select them by the mask
            # instead of assuming a prefix of the candidate axis.
            indices = [
                int(i)
                for i in torch.nonzero(level.mask[0]).reshape(-1).tolist()
            ]
            valid_indices[node_type] = indices
            if not indices:
                scores[node_type] = torch.zeros(0)
                masses[node_type] = torch.zeros(0)
                positions[node_type] = torch.zeros(0)
                continue
            head = model.detector.level_heads[node_type]
            scores[node_type] = head(values).reshape(-1).cpu()
            masses[node_type] = level.mass[0].cpu()
            positions[node_type] = level.positions[0].cpu()

    candidates = []
    for node_type in NODE_TYPES:
        for local in valid_indices[node_type]:
            # Candidate segments that carry almost no membership (mass below one
            # event equivalent) are numerical leftovers, not evidence: their
            # features are near zero and the level head scores them arbitrarily.
            if float(masses[node_type][local]) < min_mass:
                continue
            candidates.append((float(scores[node_type][local]), node_type, local))
    candidates.sort(reverse=True)
    selected = candidates[:top_k]
    selected_set = {(node_type, local) for _, node_type, local in selected}

    length = int(batch["mask"][0].sum().item())
    delta_t = batch["delta_t"][0].cpu().tolist()
    action_change = batch["action_change"][0].cpu().tolist()
    entity_change = batch["entity_change"][0].cpu().tolist()
    action_boundary = result["boundary"].action_probability[0].cpu().tolist()
    entity_boundary = result["boundary"].entity_probability[0].cpu().tolist()
    template_ids = batch["template_id"][0].cpu().tolist()
    action_ids = batch["action_id"][0].cpu().tolist()
    entity_ids = batch["entity_id"][0].cpu().tolist()
    status_ids = batch["status_id"][0].cpu().tolist()

    lookup = context.template_lookup()
    timeline = []
    for index in range(min(length, max_events)):
        timeline.append(
            {
                "event": index,
                "template_id": int(template_ids[index]),
                "template": lookup.get(int(template_ids[index]), "<UNK>"),
                "action_id": int(action_ids[index]),
                "entity_id": int(entity_ids[index]),
                "status_id": int(status_ids[index]),
                "delta_t": float(delta_t[index]),
                "action_change": float(action_change[index]),
                "entity_change": float(entity_change[index]),
                "action_boundary": float(action_boundary[index]),
                "entity_boundary": float(entity_boundary[index]),
            }
        )

    node_payload = []
    for score, node_type, local in selected:
        event_indices = node_to_events(hierarchy, node_type, local, 0, epsilon)
        node_payload.append(
            {
                "node_type": node_type,
                "local_index": local,
                "score": score,
                "mass": float(masses[node_type][local]),
                "position": float(positions[node_type][local]),
                "events": event_indices,
                "templates": top_templates(
                    context, [template_ids[i] for i in event_indices]
                ),
            }
        )

    edge_payload = []
    for edge_type in EDGE_TYPES:
        store = graph[edge_type]
        if store.edge_index.numel() == 0:
            continue
        source_type, _, target_type = edge_type
        source_attr = store.edge_index[0].cpu()
        target_attr = store.edge_index[1].cpu()
        weights = store.edge_attr[:, 0].cpu()
        keep = [
            i
            for i in range(source_attr.numel())
            if (source_type, int(source_attr[i])) in selected_set
            and (target_type, int(target_attr[i])) in selected_set
        ]
        keep.sort(key=lambda i: -float(weights[i]))
        for i in keep[:8]:
            edge_payload.append(
                {
                    "edge_type": list(edge_type),
                    "source": int(source_attr[i]),
                    "target": int(target_attr[i]),
                    "weight": float(weights[i]),
                    "distance": float(store.edge_attr[i, 1]),
                }
            )

    row = data.rows[row_index]
    return {
        "sample_id": str(row.get("sample_id", row_index)),
        "row_index": row_index,
        "label": int(row.get("label", 0)),
        "length": length,
        "anomaly_probability": float(
            detector.anomaly_probability.reshape(-1)[0].cpu()
        ),
        "anomaly_logit": float(detector.anomaly_logit.reshape(-1)[0].cpu()),
        "threshold": context.threshold,
        "level_weights": {
            name: float(detector.level_weights[0, level].cpu())
            for level, name in enumerate(NODE_TYPES)
        },
        "level_logits": {
            name: float(detector.level_logits[0, level].cpu())
            for level, name in enumerate(NODE_TYPES)
        },
        "prototype": {
            "nearest": int(detector.nearest_prototype.reshape(-1)[0].cpu()),
            "distance": float(detector.prototype_distance.reshape(-1)[0].cpu()),
        },
        "node_type_counts": {
            name: len(valid_indices[name]) for name in NODE_TYPES
        },
        "selected_nodes": node_payload,
        "subgraph_edges": edge_payload,
        "timeline": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-scan", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument(
        "--min-mass",
        type=float,
        default=1.0,
        help="ignore candidate segments whose membership mass is below this "
             "many event equivalents (they are numerical leftovers)",
    )
    parser.add_argument(
        "--cases",
        default="correct,fp,fn",
        help="comma separated subset of correct,fp,fn",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    context = load_context(args.root, args.seed, device=args.device)
    data = dataset(context, args.split)
    indices = list(range(min(args.max_scan, len(data))))
    device = default_device(args.device)

    rows: list[dict] = []
    for chunk, batch in iter_batches(data, indices, args.batch_size, device):
        result = forward_all(context.model, batch)
        probabilities = result["detector"].anomaly_probability.reshape(-1).cpu()
        labels = batch["label"].reshape(-1).cpu()
        for position, row_index in enumerate(chunk):
            rows.append(
                {
                    "row_index": row_index,
                    "label": int(labels[position]),
                    "probability": float(probabilities[position]),
                }
            )

    requested = [name.strip() for name in args.cases.split(",") if name.strip()]
    chosen, notes = select_cases(rows, requested, context.threshold)
    if not chosen:
        raise SystemExit("no case could be selected from the scanned split")

    payload = {
        "run": str(context.root),
        "seed": args.seed,
        "split": args.split,
        "scanned": len(rows),
        "threshold": context.threshold,
        "selection_notes": notes,
        "cases": {},
    }
    for name, row in chosen.items():
        case = subgraph_payload(
            context,
            data,
            row["row_index"],
            args.top_k,
            1e-5,
            args.max_events,
            args.min_mass,
        )
        case["case"] = name
        payload["cases"][name] = case
        print(
            f"[{name}] sample={case['sample_id']} label={case['label']} "
            f"p={case['anomaly_probability']:.4f} "
            f"(threshold={context.threshold}) nodes={len(case['selected_nodes'])} "
            f"edges={len(case['subgraph_edges'])}"
        )
        for node in case["selected_nodes"][:4]:
            print(
                f"    {node['node_type']:<7} score={node['score']:+.3f} "
                f"mass={node['mass']:.3f} events={node['events'][:8]} "
                f"{node['templates'][:1]}"
            )

    out_path = Path(args.out) if args.out else (
        Path(args.root) / "interpretability" / f"cases_seed{args.seed}.json"
    )
    write_json(out_path, payload)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
