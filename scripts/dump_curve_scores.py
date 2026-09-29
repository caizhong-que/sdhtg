# -*- coding: utf-8 -*-
"""dump_curve_scores.py -- score dumps for the PR / threshold-curve figure.

The unified-protocol baselines were trained from per-model configs that were
removed during the disk cleanup, so each run is rebuilt from the checkpoint's
own architecture: the TCN / Transformer / GNN-flat checkpoints load cleanly on
top of ``configs/model/sdhtg.yaml`` with only ``arch`` overridden (verified by
state-dict shape), while the ladder levels use the per-level configs saved by
``run_ladder.py``.

Usage:
    python scripts/dump_curve_scores.py --dry-run
    python scripts/dump_curve_scores.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


RUNS = [
    ("baseline_tcn", "configs/model/sdhtg.yaml", "tcn"),
    ("baseline_transformer", "configs/model/sdhtg.yaml", "transformer"),
    ("baseline_gnn_flat", "configs/model/sdhtg.yaml", "gnn_flat"),
    ("ladder_full/L0", "outputs/_ladder/main/ladder_full/model_L0.yaml", None),
    ("ladder_full/L7", "outputs/_ladder/main/ladder_full/model_L7.yaml", None),
]

PLAN = [
    ("ssh", [42, 123, 256, 512, 1024], None),
    ("hdfs", [42, 123], 20000),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=[p[0] for p in PLAN])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment["TQDM_DISABLE"] = "1"

    for dataset, seeds, max_samples in PLAN:
        if dataset not in args.datasets:
            continue
        for tag, model_config, arch in RUNS:
            targets = [
                Path(f"outputs/{dataset}/main/{tag}/seed_{seed}/test_scores.csv")
                for seed in seeds
            ]
            if all(target.is_file() for target in targets):
                print(f"== {dataset} {tag}: cached")
                continue
            command = [
                sys.executable, "scripts/dump_test_scores.py",
                "--dataset", dataset, "--tag", tag,
                "--model-config", model_config,
                "--seeds", *[str(seed) for seed in seeds],
            ]
            if arch:
                command += ["--arch", arch]
            if max_samples:
                command += ["--max-samples", str(max_samples)]
            if args.dry_run:
                print("would run:", " ".join(command))
                continue
            print(f"== {dataset} {tag}: dumping", flush=True)
            subprocess.call(command, env=environment)

    print("score dumps finished" if not args.dry_run else "[dry-run] nothing executed")


if __name__ == "__main__":
    main()
