# -*- coding: utf-8 -*-
"""Main comparison table: SDHTG (L0/L7) vs unified-input baselines, per dataset."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


METHODS = [
    ("SDHTG-L0 (平铺GRU)", "ladder_full", "L0"),
    ("SDHTG-L7 (完整)", "ladder_full", "L7"),
    ("TCN", "baseline_tcn", ""),
    ("Transformer", "baseline_transformer", ""),
    ("GNN-flat", "baseline_gnn_flat", ""),
    ("TCN-lite? (跳过)", None, None),
]


def load(dataset: str, tag: str, subdir: str) -> list[tuple[float, float]]:
    root = Path(f"outputs/{dataset}/main/{tag}")
    if subdir:
        root = root / subdir
    if not root.is_dir():
        return []
    values = []
    for seed_dir in sorted(root.glob("seed_*")):
        path = seed_dir / "result.json"
        if path.is_file():
            test = json.loads(path.read_text(encoding="utf-8"))["test"]
            values.append((test["auprc"], test["f1"]))
    return values


def fmt(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f}±{std:.4f}"


def main() -> None:
    datasets = ["bgl", "openstack", "thunderbird", "ssh", "hdfs"]
    for dataset in datasets:
        print(f"\n== {dataset} ==")
        print(f"{'方法':<22}{'AUPRC':>18}{'F1':>18}{'n':>4}")
        for label, tag, subdir in METHODS:
            if tag is None:
                continue
            values = load(dataset, tag, subdir)
            print(
                f"{label:<22}{fmt([v[0] for v in values]):>18}"
                f"{fmt([v[1] for v in values]):>18}{len(values):>4}"
            )


if __name__ == "__main__":
    main()
