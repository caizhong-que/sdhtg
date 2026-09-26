# -*- coding: utf-8 -*-
"""
run_sensitivity.py -- section 6.8 hyper-parameter sensitivity sweep.

Every group varies one manuscript hyper-parameter around the default; the
default value itself is the reference row (ladder_full/L7) and is therefore
not re-run unless --include-defaults is given.

    prototypes            K in {1,2,4,16}                     (default 8)
    prototype_temperature tau_p in {0.05,0.2,0.5}             (default 0.1)
    prototype_scale       lambda_p fixed in {0,0.25,0.5,1,2}   (default learned)
    boundary_temperature  tau_final in {0.05,0.25,0.5,1.0}    (default 0.1)
    boundary_rate         (r_A, r_E) pairs                    (default 0.05/0.01)
    temporal_radius       R_l half / double                   (default 8/4/2)
    semantic_neighbors    K_l half / double                   (default 4/4/2)
    membership_epsilon    eps_m in {1e-8,1e-4,1e-2}           (default 1e-6)

T_max is deliberately not swept here: it is a preprocessing parameter
(``max_session_length``), so changing it requires regenerating the cache. Its
cost/accuracy trade-off is covered by the efficiency table plus the chunking
note in the paper.

Finished runs are skipped, so the sweep can be interrupted and resumed, and
``--dry-run`` lists the pending runs without touching the GPU.

Usage:
    python scripts/run_sensitivity.py --dry-run
    python scripts/run_sensitivity.py --groups prototypes boundary_rate
    python scripts/run_sensitivity.py --datasets ssh --seeds 42 123 256
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


GROUPS: dict[str, list[tuple[str, list[str], list[str]]]] = {
    "prototypes": [
        ("k1", ["num_normal_prototypes=1"], []),
        ("k2", ["num_normal_prototypes=2"], []),
        ("k4", ["num_normal_prototypes=4"], []),
        ("k16", ["num_normal_prototypes=16"], []),
    ],
    "prototype_temperature": [
        ("tau005", ["prototype_temperature=0.05"], []),
        ("tau020", ["prototype_temperature=0.2"], []),
        ("tau050", ["prototype_temperature=0.5"], []),
    ],
    "prototype_scale": [
        ("lambda000", ["prototype_scale_override=0.0"], []),
        ("lambda025", ["prototype_scale_override=0.25"], []),
        ("lambda050", ["prototype_scale_override=0.5"], []),
        ("lambda100", ["prototype_scale_override=1.0"], []),
        ("lambda200", ["prototype_scale_override=2.0"], []),
    ],
    "boundary_temperature": [
        ("tau005", ["boundary.final_temperature=0.05"],
         ["curriculum.boundary_final_temperature=0.05"]),
        ("tau025", ["boundary.final_temperature=0.25"],
         ["curriculum.boundary_final_temperature=0.25"]),
        ("tau050", ["boundary.final_temperature=0.50"],
         ["curriculum.boundary_final_temperature=0.50"]),
        ("tau100", ["boundary.final_temperature=1.00"],
         ["curriculum.boundary_final_temperature=1.00"]),
    ],
    "boundary_rate": [
        ("ra001_re0005", [], ["loss.action_boundary_rate=0.01",
                              "loss.entity_boundary_rate=0.005"]),
        ("ra001_re0001", [], ["loss.action_boundary_rate=0.01",
                              "loss.entity_boundary_rate=0.001"]),
        ("ra010_re0050", [], ["loss.action_boundary_rate=0.10",
                              "loss.entity_boundary_rate=0.05"]),
        ("ra010_re0020", [], ["loss.action_boundary_rate=0.10",
                              "loss.entity_boundary_rate=0.02"]),
    ],
    "temporal_radius": [
        ("half", ["local_temporal_radius.status=4",
                  "local_temporal_radius.action=2",
                  "local_temporal_radius.entity=1"], []),
        ("double", ["local_temporal_radius.status=16",
                    "local_temporal_radius.action=8",
                    "local_temporal_radius.entity=4"], []),
    ],
    "semantic_neighbors": [
        ("half", ["semantic_neighbors.status=2",
                  "semantic_neighbors.action=2",
                  "semantic_neighbors.entity=1"], []),
        ("double", ["semantic_neighbors.status=8",
                    "semantic_neighbors.action=8",
                    "semantic_neighbors.entity=4"], []),
    ],
    "membership_epsilon": [
        ("eps1e-8", ["hierarchy.membership_epsilon=1.0e-8"], []),
        ("eps1e-4", ["hierarchy.membership_epsilon=1.0e-4"], []),
        ("eps1e-2", ["hierarchy.membership_epsilon=1.0e-2"], []),
    ],
}

PROTOTYPE_GROUPS = {"prototypes", "prototype_temperature", "prototype_scale"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=None,
                        help="subset of the parameter groups to sweep")
    parser.add_argument("--datasets", nargs="+", default=["ssh"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 256])
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prototype-stats",
        action="store_true",
        help="after each run of a prototype group, also record prototype health",
    )
    args = parser.parse_args()

    groups = args.groups or list(GROUPS)
    unknown = [name for name in groups if name not in GROUPS]
    if unknown:
        raise SystemExit(f"unknown group(s): {unknown}; known: {sorted(GROUPS)}")

    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"
    pending = 0
    completed = 0

    for dataset in args.datasets:
        config_path = Path(f"configs/experiment/{dataset}.yaml")
        if not config_path.is_file():
            print(f"!! missing experiment config {config_path}")
            continue
        output_dir = Path(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))["output_dir"]
        )
        for group in groups:
            for name, model_overrides, config_overrides in GROUPS[group]:
                tag = f"sens_{group}_{name}"
                for seed in args.seeds:
                    result_path = output_dir / f"{tag}" / f"seed_{seed}" / "result.json"
                    if result_path.is_file():
                        completed += 1
                        continue
                    if args.dry_run:
                        pending += 1
                        print(
                            f"== {dataset} {group}/{name} seed {seed}: PENDING "
                            f"(model={model_overrides} exp={config_overrides})"
                        )
                        continue
                    print(f"== {dataset} {group}/{name} seed {seed}: running",
                          flush=True)
                    command = [
                        sys.executable, "scripts/train.py",
                        "--config", str(config_path),
                        "--model-config", args.model_config,
                        "--seed", str(seed),
                        "--tag", tag,
                        "--max-epochs", str(args.max_epochs),
                        "--pretrain-epochs", "0",
                        "--skip-pretrain",
                        "--mask-template-prob", "0.0",
                    ]
                    for assignment in config_overrides:
                        command += ["--set", assignment]
                    for assignment in model_overrides:
                        command += ["--model-set", assignment]
                    code = subprocess.call(command, env=environment)
                    if code != 0:
                        print(f"!! {dataset} {group}/{name} seed {seed} failed ({code})")
                        continue
                    if args.prototype_stats and group in PROTOTYPE_GROUPS:
                        stats_command = [
                            sys.executable, "scripts/prototype_stats.py",
                            "--root", str(output_dir / tag),
                            "--seed", str(seed),
                            "--split", "train",
                            "--max-samples", "5000",
                        ]
                        subprocess.call(stats_command, env=environment)

    if args.dry_run:
        print(
            f"[dry-run] sensitivity: {pending} runs pending, "
            f"{completed} already finished"
        )
    else:
        print(f"sensitivity sweep finished ({completed} runs were already cached)")


if __name__ == "__main__":
    main()
