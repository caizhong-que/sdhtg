# -*- coding: utf-8 -*-
"""Inventory of completed runs per group/dataset/variant (to avoid re-runs)."""
from __future__ import annotations

from pathlib import Path


GROUPS = (
    "ladder_full",
    "baseline_tcn",
    "baseline_transformer",
    "baseline_gnn_flat",
    "ablation_shortcuts",
    "ablation_boundary",
    "ablation_imbalance",
    "ablation_candidates",
    "pretrain",
    "lf_",
)


def main() -> None:
    for dataset in ("hdfs", "ssh", "bgl", "openstack", "thunderbird",
                    "bgl_iter", "bgl_synth"):
        root = Path(f"outputs/{dataset}/main")
        if not root.is_dir():
            continue
        rows = []
        for tag in sorted(root.iterdir()):
            if not tag.is_dir():
                continue
            if not any(tag.name.startswith(prefix) for prefix in GROUPS):
                continue
            count = len(list(tag.rglob("result.json")))
            if count:
                rows.append((tag.name, count))
        if not rows:
            continue
        print(f"== {dataset} ==")
        for name, count in rows:
            print(f"   {name:<46}{count:>4} runs")


if __name__ == "__main__":
    main()
