# -*- coding: utf-8 -*-
"""
run_parsing_noise.py -- section 6.5 parsing-noise robustness grid.

Four parser-corruption kinds x two contamination protocols x two seeds = 16
runs on one dataset (default SSH, where the main model shows its advantage):

    replace / merge / split / unk   x   test_only / all   x   seeds 42 123

Each variant first materialises a corrupted cache (``make_noisy_dataset.py``,
cached on disk) and then trains the main model on it. Finished runs are
skipped, so the grid can be interrupted and resumed; ``--dry-run`` lists the
work without touching the GPU.

Usage:
    python scripts/run_parsing_noise.py --dry-run
    python scripts/run_parsing_noise.py --datasets ssh --seeds 42 123 --rate 0.2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


KINDS = ("replace", "merge", "split", "unk")
PROTOCOLS = ("test_only", "all")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ssh"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--kinds", nargs="+", default=list(KINDS))
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"
    pending = 0
    finished = 0

    for dataset in args.datasets:
        config_path = Path(f"configs/experiment/{dataset}.yaml")
        if not config_path.is_file():
            print(f"!! missing {config_path}")
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        source = Path(config["data"]["processed_dir"])
        output_dir = Path(config["output_dir"]) / f"noise_r{args.rate:g}"

        for kind in args.kinds:
            for protocol in args.protocols:
                noise_dir = source.with_name(f"{source.name}_noise_{kind}_{protocol}")
                if not noise_dir.is_dir() or args.rebuild_data:
                    if args.dry_run:
                        print(f"== would build {noise_dir}")
                    else:
                        print(f"== building {noise_dir}", flush=True)
                        subprocess.call(
                            [
                                sys.executable, "scripts/make_noisy_dataset.py",
                                "--source", str(source),
                                "--kind", kind,
                                "--protocol", protocol,
                                "--rate", str(args.rate),
                                *(["--overwrite"] if args.rebuild_data else []),
                            ],
                            env=environment,
                        )
                tag = f"{kind}_{protocol}"
                for seed in args.seeds:
                    result = output_dir / tag / f"seed_{seed}" / "result.json"
                    if result.is_file():
                        finished += 1
                        continue
                    if args.dry_run:
                        pending += 1
                        print(f"== {dataset} {tag} seed {seed}: PENDING")
                        continue
                    print(f"== {dataset} {tag} seed {seed}: running", flush=True)
                    command = [
                        sys.executable, "scripts/train.py",
                        "--config", str(config_path),
                        "--set", f"data.processed_dir={noise_dir.as_posix()}",
                        "--set", f"output_dir={output_dir.as_posix()}",
                        "--model-config", args.model_config,
                        "--seed", str(seed),
                        "--tag", tag,
                        "--max-epochs", str(args.max_epochs),
                        "--pretrain-epochs", "0",
                        "--skip-pretrain",
                        "--mask-template-prob", "0.0",
                    ]
                    code = subprocess.call(command, env=environment)
                    if code != 0:
                        print(f"!! {dataset} {tag} seed {seed} failed ({code})")

    if args.dry_run:
        print(f"[dry-run] parsing noise: {pending} runs pending, {finished} finished")
    else:
        print(f"parsing-noise grid finished ({finished} runs were already cached)")


if __name__ == "__main__":
    main()
