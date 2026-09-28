# -*- coding: utf-8 -*-
"""run_label_scarcity.py -- label-fraction protocol for a second dataset.

Section 6.4 (RQ4) currently reports label scarcity on a BGL subset only.  This
driver repeats the protocol on another dataset (default SSH, the dataset with
the largest dynamic range) for the flat baseline (L0) and the full model (L7),
using exactly the ladder level definitions from ``run_ladder.py``.

Unlabelled samples keep their position in the loader (they still take part in
the boundary regulariser) but contribute no supervised loss, matching the
protocol described in Section 5.5.

Usage:
    python scripts/run_label_scarcity.py --dry-run
    python scripts/run_label_scarcity.py --dataset ssh --fractions 0.01 0.05 0.1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ladder import LEVEL_ABLATIONS, level_model_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ssh")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--levels", nargs="+", default=["L0", "L7"])
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.01, 0.05, 0.10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 256])
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--tag", default="label_scarcity")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config or f"configs/experiment/{args.dataset}.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    scratch = Path(args.output_root) / "_ladder" / output_dir.name / args.tag
    scratch.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")

    for level in args.levels:
        if level not in LEVEL_ABLATIONS:
            raise SystemExit(f"unknown level {level!r}")
        model_path = scratch / f"model_{level}.yaml"
        model_path.write_text(
            yaml.safe_dump({"model": level_model_config(args.model_config, level)},
                           allow_unicode=True),
            encoding="utf-8",
        )
        for fraction in args.fractions:
            tag = f"{args.tag}/{level}_lf{int(round(fraction * 100)):02d}"
            for seed in args.seeds:
                result = output_dir / tag / f"seed_{seed}" / "result.json"
                if result.is_file():
                    print(f"  {tag} seed {seed}: cached")
                    continue
                command = [
                    sys.executable, "scripts/train.py",
                    "--config", str(config_path),
                    "--model-config", str(model_path),
                    "--seed", str(seed), "--tag", tag,
                    "--max-epochs", str(args.max_epochs),
                    "--pretrain-epochs", "0", "--mask-template-prob", "0.0",
                    "--skip-pretrain",
                    "--label-fraction", str(fraction),
                ]
                if args.dry_run:
                    print("  would run:", " ".join(command))
                    continue
                print(f"  {tag} seed {seed}: running", flush=True)
                code = subprocess.call(command, env=environment)
                if code != 0:
                    print(f"  !! {tag} seed {seed} failed ({code})")

    print("label-scarcity protocol finished"
          if not args.dry_run else "[dry-run] nothing executed")


if __name__ == "__main__":
    main()
