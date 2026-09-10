# -*- coding: utf-8 -*-
"""Print a compact progress table for ladders, baselines and ablations."""
import json
import statistics
from pathlib import Path

GROUPS = {
    "ladder_full": ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L7b"],
    "baseline_tcn": [""],
    "baseline_transformer": [""],
    "baseline_gnn_flat": [""],
}


def summarise(output_dir: Path, tag: str, levels: list[str]) -> None:
    root = output_dir / tag
    if not root.is_dir():
        return
    rows = []
    for level in levels:
        level_dir = root / level if level else root
        values = []
        for seed_dir in sorted(level_dir.glob("seed_*")):
            path = seed_dir / "result.json"
            if path.is_file():
                test = json.loads(path.read_text(encoding="utf-8"))["test"]
                values.append((test["auprc"], test["f1"]))
        if values:
            rows.append(
                (
                    level or "-",
                    len(values),
                    statistics.mean(v[0] for v in values),
                    statistics.mean(v[1] for v in values),
                )
            )
    if not rows:
        return
    print(f"  {tag}:")
    for level, count, auprc, f1 in rows:
        print(f"    {level:<5} n={count}  AUPRC={auprc:.4f}  F1={f1:.4f}")


def main() -> None:
    datasets = ["openstack", "ssh", "bgl", "hdfs", "thunderbird"]
    for dataset in datasets:
        output_dir = Path(f"outputs/{dataset}/main")
        if not output_dir.is_dir():
            continue
        print(f"== {dataset} ==")
        for tag, levels in GROUPS.items():
            summarise(output_dir, tag, levels)
        for ablation_dir in sorted(output_dir.glob("ablation_*")):
            summarise(output_dir, ablation_dir.name, [""])


if __name__ == "__main__":
    main()
