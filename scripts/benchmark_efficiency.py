"""Measure per-sequence-length training/inference time, memory, and graph size.

Usage:
    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --lengths 32,64 --batch-sizes 2048,1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sdhtg.models.factory import build_model


DEFAULT_LENGTHS = [32, 64, 128, 256, 512]
DEFAULT_BATCH_SIZES = [2048, 1024, 512, 256, 128]


def synthetic_batch(batch_size: int, steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    mask = torch.ones(batch_size, steps, dtype=torch.bool, device=device)
    return {
        "template_id": torch.randint(2, 512, (batch_size, steps), device=device),
        "entity_id": torch.randint(2, 256, (batch_size, steps), device=device),
        "action_id": torch.randint(2, 64, (batch_size, steps), device=device),
        "status_id": torch.randint(2, 512, (batch_size, steps), device=device),
        "delta_t": torch.rand(batch_size, steps, device=device),
        "action_change": torch.randint(0, 2, (batch_size, steps), device=device).float(),
        "entity_change": torch.randint(0, 2, (batch_size, steps), device=device).float(),
        "mask": mask,
    }


def count_edges(graph) -> int:
    total = 0
    for store in graph.edge_stores:
        total += int(store.edge_index.shape[1]) if store.edge_index is not None else 0
    return total


def measure(
    model: torch.nn.Module,
    batch_size: int,
    steps: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    batch = synthetic_batch(batch_size, steps, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        out.anomaly_logit.mean().backward()
        # No parameter update: benchmark measures forward+backward cost only,
        # which avoids training-divergence NaN in random synthetic batches.

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    train_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        out.anomaly_logit.mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_times.append((time.perf_counter() - start) * 1000)

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            out = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            out = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times.append((time.perf_counter() - start) * 1000)

    peak_gb = (
        float(torch.cuda.max_memory_allocated()) / 1024**3
        if device.type == "cuda"
        else float("nan")
    )
    return {
        "T": steps,
        "batch_size": batch_size,
        "train_ms": sum(train_times) / len(train_times),
        "infer_ms": sum(infer_times) / len(infer_times),
        "peak_gb": peak_gb,
        "edges": count_edges(out.graph_batch),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if len(lengths) != len(batch_sizes):
        raise ValueError("--lengths and --batch-sizes must have the same length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = build_model(args.model_config).to(device)

    rows = []
    for steps, batch_size in zip(lengths, batch_sizes):
        row = measure(
            model, batch_size, steps, device, args.warmup, args.repeats
        )
        rows.append(row)
        print(
            f"T={row['T']:>4d}  batch={row['batch_size']:>5d}  "
            f"train={row['train_ms']:8.2f} ms  infer={row['infer_ms']:8.2f} ms  "
            f"peak={row['peak_gb']:6.2f} GB  edges={row['edges']:>8d}"
        )

    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
