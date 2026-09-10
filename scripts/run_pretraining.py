"""
run_pretraining.py -- contrastive pretraining matrix (paper Table 10).

Compares pretraining protocols (normal-only vs all-train) x negative-sample
strategies (random / hard / semi-hard / semantic-filtered / none / supervised),
each followed by the standard supervised fine-tuning stage. The no-pretraining
reference is the ladder's L7 run.

Usage:
    python scripts/run_pretraining.py
    python scripts/run_pretraining.py --datasets ssh --seeds 42 123 --pretrain-epochs 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


STRATEGIES = ("random", "hard", "semi_hard", "semantic", "none", "supervised")
PROTOCOLS = ("normal_only", "all_train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ssh"])
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 256])
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument(
        "--model-config", default="configs/model/sdhtg.yaml",
        help="model configuration used for pretraining + fine-tuning",
    )
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"

    for dataset in args.datasets:
        config_path = Path(f"configs/experiment/{dataset}.yaml")
        if not config_path.is_file():
            print(f"!! missing config {config_path}")
            continue
        output_dir = Path(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))["output_dir"]
        )
        for protocol in args.protocols:
            for strategy in args.strategies:
                tag = f"pretrain_{protocol}_{strategy}"
                for seed in args.seeds:
                    if (output_dir / f"{tag}/seed_{seed}/result.json").is_file():
                        print(f"== {dataset} {protocol}/{strategy} seed {seed}: cached")
                        continue
                    print(
                        f"== {dataset} {protocol}/{strategy} seed {seed}: running",
                        flush=True,
                    )
                    code = subprocess.call(
                        [
                            sys.executable, "scripts/train.py",
                            "--config", str(config_path),
                            "--model-config", args.model_config,
                            "--seed", str(seed),
                            "--tag", tag,
                            "--max-epochs", str(args.max_epochs),
                            "--pretrain-epochs", str(args.pretrain_epochs),
                            "--pretrain-protocol", protocol,
                            "--negative-strategy", strategy,
                            "--mask-template-prob", "0.0",
                        ],
                        env=environment,
                    )
                    if code != 0:
                        print(f"!! {dataset} {protocol}/{strategy} seed {seed} failed ({code})")
    print("pretraining matrix finished")


if __name__ == "__main__":
    main()
