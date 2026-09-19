# -*- coding: utf-8 -*-
"""Compare candidate main models against the full ladder L7 model."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def load(dataset: str, tag: str, subdir: str = "") -> list[tuple[float, float]]:
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
    for dataset in ("ssh", "hdfs"):
        print(f"== {dataset} ==")
        for label, tag, subdir in (
            ("完整模型 (ladder L7)", "ladder_full", "L7"),
            ("候选: 层次+原型-无图边", "ablation_candidates_nograph", ""),
        ):
            values = load(dataset, tag, subdir)
            print(
                f"  {label:<26}{fmt([v[0] for v in values]):>18}"
                f"{fmt([v[1] for v in values]):>18}  n={len(values)}"
            )


if __name__ == "__main__":
    main()
