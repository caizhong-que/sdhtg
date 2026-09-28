# -*- coding: utf-8 -*-
"""run_entity_holdout.py -- sequential driver for the seen/unseen entity study.

Stages, all resumable (finished artefacts are skipped):

    1. build the entity-holdout cache  (make_entity_holdout.py)
    2. train the main model on it      (train.py, ladder settings)
    3. grouped evaluation              (evaluate_seen_unseen.py)

Training is strictly sequential: running two jobs at once on the 8 GB card
pushes the allocator into shared-memory fallback and slows every step by an
order of magnitude (observed 2026-09-28), which is exactly what the earlier
batch file did by accident.

Usage:
    python scripts/run_entity_holdout.py --dry-run
    python scripts/run_entity_holdout.py --datasets bgl openstack --seeds 42 123
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["bgl", "openstack"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--coverage", type=float, default=0.2)
    parser.add_argument("--level", default="L7")
    parser.add_argument("--tag", default="entholdout20")
    parser.add_argument("--max-seen", type=int, default=20000,
                        help="cap on the seen group during evaluation (0 = no cap)")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--fast-datasets", nargs="+", default=["bgl"],
                        help="datasets that get --fast-epochs (converged early on the ladder)")
    parser.add_argument("--fast-epochs", type=int, default=8)
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment["TQDM_DISABLE"] = "0"

    def call(command: list[str]) -> int:
        if args.dry_run:
            print("  would run:", " ".join(command))
            return 0
        print("  run:", " ".join(command), flush=True)
        return subprocess.call(command, env=environment)

    for dataset in args.datasets:
        config = yaml.safe_load(
            Path(f"configs/experiment/{dataset}.yaml").read_text(encoding="utf-8"))
        source = Path(config["data"]["processed_dir"])
        cache = source.with_name(f"{source.name}_{args.tag}")
        output_dir = Path(config["output_dir"]) / args.tag

        print(f"== {dataset}: cache={cache}")
        if not (cache / "sessions.parquet").is_file():
            call([sys.executable, "scripts/make_entity_holdout.py",
                  "--source", str(source), "--coverage", str(args.coverage)])
        else:
            print("  cache exists")

        for seed in args.seeds:
            result = output_dir / args.level / f"seed_{seed}" / "result.json"
            if result.is_file():
                print(f"  seed {seed}: cached")
                continue
            epochs = args.fast_epochs if dataset in args.fast_datasets else args.max_epochs
            print(f"  seed {seed}: training ({epochs} epochs)", flush=True)
            code = call([
                sys.executable, "scripts/train.py",
                "--config", f"configs/experiment/{dataset}.yaml",
                "--set", f"data.processed_dir={cache.as_posix()}",
                "--set", f"output_dir={output_dir.as_posix()}",
                "--model-config", args.model_config,
                "--seed", str(seed), "--tag", args.level,
                "--max-epochs", str(epochs),
                "--pretrain-epochs", "0", "--mask-template-prob", "0.0",
                "--skip-pretrain",
            ])
            if code != 0:
                print(f"  !! seed {seed} failed ({code})")

        print(f"  evaluating {dataset}", flush=True)
        evaluate = [
            sys.executable, "scripts/evaluate_seen_unseen.py",
            "--datasets", dataset, "--seeds", *[str(s) for s in args.seeds],
            "--ladder", args.tag, "--level", args.level,
            "--data-dir", cache.as_posix(),
            "--groups-file", (cache / "holdout_groups.csv").as_posix(),
        ]
        if args.max_seen:
            evaluate += ["--max-seen", str(args.max_seen)]
        call(evaluate)

    print("entity-holdout study finished"
          if not args.dry_run else "[dry-run] nothing executed")


if __name__ == "__main__":
    main()
