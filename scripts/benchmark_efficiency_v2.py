"""
benchmark_efficiency_v2.py -- efficiency comparison across sequence lengths
for SDHTG and the comparison models (paper Table 11 / RQ6).

Reports, for each (model, length, batch size): training step time (ms),
inference time per sample (ms), peak GPU memory (GB) and, for graph models,
the number of instantiated graph edges.

Usage:
    python scripts/benchmark_efficiency_v2.py
    python scripts/benchmark_efficiency_v2.py --models sdhtg transformer --lengths 64,256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sdhtg.models.factory import build_model


MODEL_CONFIGS = {
    "sdhtg": "configs/model/sdhtg.yaml",
    "gru_flat": "configs/model/ladder_l0.yaml",
    "tcn": "configs/model/tcn.yaml",
    "transformer": "configs/model/transformer.yaml",
    "gnn_flat": "configs/model/gnn_flat.yaml",
}

# Batch sizes follow the training-time bucket sampler (short -> large batch).
DEFAULT_LENGTHS = [32, 64, 128, 256, 512]
DEFAULT_BATCH_SIZES = [512, 128, 64, 64, 64]


def synthetic_batch(batch_size: int, steps: int, device: torch.device) -> dict:
    return {
        "template_id": torch.randint(2, 400, (batch_size, steps), device=device),
        "entity_id": torch.randint(2, 300, (batch_size, steps), device=device),
        "action_id": torch.randint(2, 30, (batch_size, steps), device=device),
        "status_id": torch.randint(2, 200, (batch_size, steps), device=device),
        "delta_t": torch.rand(batch_size, steps, device=device),
        "action_change": torch.randint(0, 2, (batch_size, steps), device=device).float(),
        "entity_change": torch.randint(0, 2, (batch_size, steps), device=device).float(),
        "mask": torch.ones(batch_size, steps, dtype=torch.bool, device=device),
    }


def graph_edges(output) -> int:
    graph = getattr(output, "graph_batch", None)
    if graph is None:
        return 0
    total = 0
    for store in getattr(graph, "edge_stores", []):
        index = getattr(store, "edge_index", None)
        if index is not None:
            total += int(index.shape[1])
    return total


def measure(
    model: torch.nn.Module,
    batch_size: int,
    steps: int,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> dict:
    batch = synthetic_batch(batch_size, steps, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        output.anomaly_logit.mean().backward()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    train_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = model(batch)
        output.anomaly_logit.mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_times.append((time.perf_counter() - start) * 1000)
        optimizer.zero_grad(set_to_none=True)
    peak = (
        torch.cuda.max_memory_allocated() / 1024**3
        if device.type == "cuda"
        else float("nan")
    )

    model.eval()
    edges = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_times = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = time.perf_counter()
            output = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times.append((time.perf_counter() - start) * 1000)
            edges = graph_edges(output)

    return {
        "train_ms": sum(train_times) / len(train_times),
        "infer_ms_per_sample": (sum(infer_times) / len(infer_times)) / batch_size,
        "peak_memory_gb": peak,
        "graph_edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS))
    parser.add_argument(
        "--lengths", nargs="+", default=[str(x) for x in DEFAULT_LENGTHS],
        help="sequence lengths; accepts '32,64' or '32 64'",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+",
        default=[str(x) for x in DEFAULT_BATCH_SIZES],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="outputs/efficiency_benchmark.json")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="reuse (model, length) entries already present in --out",
    )
    args = parser.parse_args()

    def to_ints(values: list[str]) -> list[int]:
        return [int(x) for chunk in values for x in str(chunk).split(",") if x]

    lengths = to_ints(args.lengths)
    batch_sizes = to_ints(args.batch_sizes)
    assert len(lengths) == len(batch_sizes), "lengths and batch sizes must pair up"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[str, dict] = {}
    out_path = Path(args.out)
    if args.skip_existing and out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        results = {
            model: {int(k): v for k, v in entries.items()}
            for model, entries in results.items()
        }
        print(f"reusing {out_path} ({sum(len(v) for v in results.values())} entries)")

    for name in args.models:
        config_path = MODEL_CONFIGS.get(name)
        if config_path is None or not Path(config_path).is_file():
            print(f"!! unknown model {name}")
            continue
        for steps, batch_size in zip(lengths, batch_sizes):
            if steps in results.get(name, {}):
                print(f"{name:<12} T={steps:<4} cached")
                continue
            try:
                model = build_model(config_path).to(device)
                metrics = measure(
                    model, batch_size, steps, device, args.warmup, args.repeats
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                metrics = {"train_ms": float("nan"), "infer_ms_per_sample": float("nan"),
                           "peak_memory_gb": float("nan"), "graph_edges": 0,
                           "oom": True}
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    metrics = {"train_ms": float("nan"), "infer_ms_per_sample": float("nan"),
                               "peak_memory_gb": float("nan"), "graph_edges": 0,
                               "oom": True}
                else:
                    raise
            results.setdefault(name, {})[steps] = metrics
            print(
                f"{name:<12} T={steps:<4} B={batch_size:<5} "
                f"train={metrics['train_ms']:>9.1f} ms  "
                f"infer={metrics['infer_ms_per_sample']:>7.3f} ms/样本  "
                f"peak={metrics['peak_memory_gb']:>5.2f} GB  "
                f"edges={metrics['graph_edges']}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
