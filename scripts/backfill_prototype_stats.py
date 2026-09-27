# -*- coding: utf-8 -*-
"""
backfill_prototype_stats.py -- fill in prototype health metrics that are missing.

The first post-pretraining pipeline run hit a CUDA OOM in prototype_stats.py
(SSH sessions average ~280 events, so a fixed 256-sample batch built a graph
that does not fit in 8 GB). The batch construction is fixed now, but runs whose
stats were lost must be revisited. This script scans the runs of a dataset for
``seed_<s>/result.json`` without ``interpretability/prototype_stats_seed<s>.json``
and re-runs the statistics for them.

Usage:
    python scripts/backfill_prototype_stats.py --datasets ssh --seeds 42 123 256
    python scripts/backfill_prototype_stats.py --datasets ssh --only sens_prototypes
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TAGS = (
    "ladder_full/L7",
    "sens_prototypes_",
    "sens_prototype_temperature_",
    "sens_prototype_scale_",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ssh"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 256, 512, 1024])
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS),
                        help="tag prefixes (relative to outputs/<ds>/main) to scan")
    parser.add_argument("--only", default=None,
                        help="restrict to tags containing this substring")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"
    pending = 0
    for dataset in args.datasets:
        main_dir = Path("outputs") / dataset / "main"
        if not main_dir.is_dir():
            print(f"!! missing {main_dir}")
            continue
        for prefix in args.tags:
            base = main_dir / prefix
            candidates = [base] if (base / "seed_42").is_dir() else sorted(
                p for p in main_dir.glob(f"{prefix}*") if p.is_dir()
            )
            for run_dir in candidates:
                if args.only and args.only not in run_dir.name:
                    continue
                for seed in args.seeds:
                    seed_dir = run_dir / f"seed_{seed}"
                    if not (seed_dir / "result.json").is_file():
                        continue
                    target = (
                        run_dir / "interpretability" /
                        f"prototype_stats_seed{seed}.json"
                    )
                    if target.is_file():
                        continue
                    pending += 1
                    if args.dry_run:
                        print(f"== pending: {run_dir.name} seed {seed}")
                        continue
                    print(f"== {run_dir.name} seed {seed}: running", flush=True)
                    command = [
                        sys.executable, "scripts/prototype_stats.py",
                        "--root", str(run_dir),
                        "--seed", str(seed),
                        "--split", args.split,
                        "--max-samples", str(args.max_samples),
                    ]
                    if args.device:
                        command += ["--device", args.device]
                    code = subprocess.call(command, env=environment)
                    if code != 0:
                        print(f"!! {run_dir.name} seed {seed} failed ({code})")

    if args.dry_run:
        print(f"[dry-run] {pending} prototype-stats runs pending")
    else:
        print(f"backfill finished ({pending} runs processed)")


if __name__ == "__main__":
    main()
