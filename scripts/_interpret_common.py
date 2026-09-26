# -*- coding: utf-8 -*-
"""
_interpret_common.py -- shared helpers for the section 6.6 evidence scripts.

The four interpretability scripts (boundaries, hierarchy, prototypes, cases)
all need the same three things: locate a finished run and load its checkpoint,
run the model while keeping the intermediate tensors, and map graph nodes back
to the events they were built from. That logic lives here so the four scripts
stay short and consistent.

Nothing in this module trains or writes checkpoints; it is read-only with
respect to ``outputs/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
import yaml

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.models.graph import GraphEncoderOutput, NODE_TYPES


def default_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunContext:
    """Everything needed to reproduce a finished run's forward pass."""

    root: Path
    seed: int
    model: torch.nn.Module
    device: torch.device
    processed: Path
    vocab: dict[str, dict[str, int]]
    threshold: float | None
    model_config: Path | None

    @property
    def dataset_name(self) -> str:
        parts = self.root.parts
        return parts[parts.index("outputs") + 1] if "outputs" in parts else "unknown"

    @property
    def tag(self) -> str:
        return self.root.name

    def template_lookup(self) -> dict[int, str]:
        return {int(v): k for k, v in self.vocab["template"].items()}


def _manifest_paths(manifest: dict[str, Any]) -> tuple[Path | None, Path | None]:
    """Recover the processed-data dir and the model config from a manifest.

    Two conventions exist in this project: ablation runs record
    ``configs/model/<name>.yaml``, while ladder runs record a generated config
    such as ``outputs/_ladder/main/ladder_full/model_L7.yaml``. The experiment
    config is identified by its ``configs/experiment/`` prefix, so whatever
    remains is the model config.
    """
    processed = None
    for raw in manifest.get("data") or {}:
        if str(raw).endswith("sessions.parquet"):
            processed = Path(raw).parent
            break

    candidates: list[str] = []
    for raw in manifest.get("configs") or {}:
        normalized = str(raw).replace("\\", "/")
        if normalized.startswith("configs/experiment/"):
            continue
        candidates.append(normalized)

    model_config = None
    for candidate in candidates:
        if Path(candidate).is_file():
            model_config = Path(candidate)
            break
    if model_config is None and candidates:
        model_config = Path(candidates[0])
    return processed, model_config


def load_context(
    root: str | Path,
    seed: int,
    *,
    device: str | None = None,
    model_config: str | None = None,
    processed_dir: str | None = None,
) -> RunContext:
    """Load a finished run (``outputs/<ds>/main/<tag>``) and its checkpoint."""
    root = Path(root)
    seed_dir = root / f"seed_{seed}"
    manifest_path = seed_dir / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing training manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_processed, manifest_model = _manifest_paths(manifest)

    processed = Path(processed_dir) if processed_dir else manifest_processed
    if processed is None:
        raise ValueError("could not infer processed_dir; pass --processed-dir")
    resolved_model_config = model_config or manifest_model
    if resolved_model_config is None:
        raise ValueError("could not infer the model config; pass --model-config")

    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    overrides: dict[str, Any] = {
        f"{name}_vocab_size": len(vocab[name])
        for name in ("template", "entity", "action", "status")
    }
    # Sensitivity runs vary the model config through CLI overrides; replay them
    # so the reloaded architecture matches the trained weights.
    override_record = seed_dir / "cli_overrides.json"
    if override_record.is_file():
        recorded = json.loads(override_record.read_text(encoding="utf-8"))
        for assignment in recorded.get("model_set", []):
            key, _, raw = assignment.partition("=")
            overrides.setdefault(key.strip(), yaml.safe_load(raw))

    device_obj = default_device(device)
    model = build_model(resolved_model_config, overrides).to(device_obj).eval()
    checkpoint = torch.load(
        seed_dir / "checkpoints" / "best.pt",
        map_location=device_obj,
        weights_only=False,
    )
    state = checkpoint.get("model", checkpoint)
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)

    threshold = None
    threshold_path = seed_dir / "threshold.json"
    if threshold_path.is_file():
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        if "threshold" in payload:
            threshold = float(payload["threshold"])

    return RunContext(
        root=root,
        seed=seed,
        model=model,
        device=device_obj,
        processed=processed,
        vocab=vocab,
        threshold=threshold,
        model_config=Path(resolved_model_config),
    )


def dataset(context: RunContext, split: str) -> SessionDataset:
    return SessionDataset(str(context.processed / "sessions.parquet"), split)


def iter_batches(
    data: SessionDataset,
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[list[int], dict[str, torch.Tensor]]]:
    """Yield ``(row_indices, device_batch)``; row order is preserved."""
    # Long sessions (SSH averages ~280 events) make the graph tensors large: a
    # 256-sample CPU batch can ask the host allocator for multi-gigabyte
    # intermediates, so cap the CPU batch size.
    effective = min(batch_size, 32) if device.type == "cpu" else batch_size
    for start in range(0, len(indices), effective):
        chunk = list(indices[start : start + effective])
        rows = [data[i] for i in chunk]
        yield chunk, move_batch_to_device(collate_sessions(rows), device)


def forward_all(model, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Run SDHTG module by module, keeping the section 6.6 evidence tensors."""
    with torch.no_grad():
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
        detector_output = model.detector(
            GraphEncoderOutput(
                graph=graph_output.graph,
                node_embeddings=graph_output.node_embeddings,
            ),
            strategy_output.sequence_strategy,
        )
    return {
        "batch": batch,
        "event": event_output,
        "strategy": strategy_output,
        "boundary": boundary_output,
        "hierarchy": hierarchy_output,
        "graph_build": graph_build,
        "graph": graph_output,
        "detector": detector_output,
    }


def forward_boundary(model, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Encoder + strategy FiLM + boundary network only.

    Boundary-prior checks do not need the hierarchy or the graph, and graph
    construction dominates the runtime, so this keeps such scans cheap.
    """
    with torch.no_grad():
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
    return {"boundary": boundary_output}


def node_scores(model, graph_output) -> dict[str, torch.Tensor]:
    """Per-node anomaly evidence: the level-head logit of each node."""
    scores: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for node_type in NODE_TYPES:
            values = graph_output.node_embeddings[node_type]
            if values.numel() == 0:
                scores[node_type] = values.reshape(-1)
                continue
            head = model.detector.level_heads[node_type]
            scores[node_type] = head(values).reshape(-1)
    return scores


def node_to_events(
    hierarchy,
    node_type: str,
    index: int,
    sample: int,
    epsilon: float = 1e-5,
) -> list[int]:
    """Map a graph node back to the events it was aggregated from.

    ``sample`` selects the row inside the batch; the returned values are event
    offsets inside that outer sample (0-based, causal order).
    """
    if node_type == "status":
        return [int(index)]
    if node_type == "action":
        membership = hierarchy.status_to_action[sample]
        weights = membership[:, int(index)]
        return [int(i) for i in torch.nonzero(weights > epsilon).reshape(-1).tolist()]
    if node_type == "entity":
        action_membership = hierarchy.action_to_entity[sample]
        action_indices = torch.nonzero(
            action_membership[:, int(index)] > epsilon
        ).reshape(-1)
        events: list[int] = []
        for action_index in action_indices.tolist():
            events.extend(
                node_to_events(hierarchy, "action", action_index, sample, epsilon)
            )
        return sorted(set(events))
    raise ValueError(f"unknown node type {node_type!r}")


def template_words(context: RunContext, ids: Iterable[int]) -> list[str]:
    lookup = context.template_lookup()
    return [lookup.get(int(i), f"<{int(i)}>") for i in ids]


def top_templates(context: RunContext, ids: Iterable[int], top: int = 3) -> list[str]:
    counts: dict[int, int] = {}
    for raw in ids:
        value = int(raw)
        if value <= 1:  # PAD / UNK
            continue
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    lookup = context.template_lookup()
    return [lookup.get(key, f"<{key}>") for key, _ in ordered]


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def default_out(root: Path, name: str, seed: int) -> Path:
    return root / "interpretability" / f"{name}_seed{seed}.json"
