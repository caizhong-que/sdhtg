"""
run_baselines.py -- unified-input-protocol baselines (TCN / Transformer /
flat graph) across the five datasets and five seeds.

Resumes automatically: runs whose result.json already exists are skipped.

Usage:
    python scripts/run_baselines.py                # all datasets, all models
    python scripts/run_baselines.py --datasets ssh hdfs
    python scripts/run_baselines.py --models tcn
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


MODELS = ("tcn", "transformer", "gnn_flat")
SEEDS = (42, 123, 256, 512, 1024)


def already_done(output_dir: Path, tag: str, seed: int) -> bool:
    return (output_dir / f"{tag}/seed_{seed}/result.json").is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=["ssh", "openstack", "bgl", "hdfs", "thunderbird"],
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=30)
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
        for model in args.models:
            model_config = Path(f"configs/model/{model}.yaml")
            if not model_config.is_file():
                print(f"!! missing model config {model_config}")
                continue
            tag = f"baseline_{model}"
            for seed in args.seeds:
                if already_done(output_dir, tag, seed):
                    print(f"== {dataset} {model} seed {seed}: cached")
                    continue
                print(f"== {dataset} {model} seed {seed}: running", flush=True)
                code = subprocess.call(
                    [
                        sys.executable, "scripts/train.py",
                        "--config", str(config_path),
                        "--model-config", str(model_config),
                        "--seed", str(seed),
                        "--tag", tag,
                        "--max-epochs", str(args.max_epochs),
                        "--pretrain-epochs", "0",
                        "--skip-pretrain",
                        "--mask-template-prob", "0.0",
                    ],
                    env=environment,
                )
                if code != 0:
                    print(f"!! {dataset} {model} seed {seed} failed ({code})")

    print("baseline runs finished")


if __name__ == "__main__":
    main()
