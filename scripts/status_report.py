# -*- coding: utf-8 -*-
"""Compact status: finished result sets per phase."""
from pathlib import Path


def count_results(path: Path) -> int:
    return len(list(path.rglob("result.json"))) if path.is_dir() else 0


def main() -> None:
    root = Path("outputs")
    datasets = ["bgl", "openstack", "thunderbird", "ssh", "hdfs"]
    print("== baselines ==")
    for model in ("tcn", "transformer", "gnn_flat"):
        per_dataset = {
            ds: count_results(root / ds / "main" / f"baseline_{model}")
            for ds in datasets
        }
        print(f"  baseline_{model}: {per_dataset}")
    print("== ladders ==")
    for ds in datasets:
        print(f"  {ds}: {count_results(root / ds / 'main' / 'ladder_full')}/45")
    print("== ablations ==")
    for path in sorted(root.rglob("ablation_*")):
        if path.is_dir():
            print(f"  {path.relative_to(root)}: {count_results(path)}")
    print("== pretraining ==")
    for path in sorted(root.rglob("pretrain_*")):
        if path.is_dir():
            print(f"  {path.relative_to(root)}: {count_results(path)}")
    print("== efficiency ==")
    for name in ("efficiency_benchmark.json", "_smoke_efficiency.json", "_smoke_eff2.json"):
        path = root / name
        print(f"  {name}: {'present' if path.is_file() else 'missing'}")


if __name__ == "__main__":
    main()
