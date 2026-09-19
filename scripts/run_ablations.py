"""
run_ablations.py -- P1 ablation experiments in three groups.

    shortcuts  input-semantics and entity-shortcut ablations   (paper Table 9)
    imbalance  class-imbalance loss / prototype ablations      (paper Table 8)
    boundary   boundary source and cross-level gradient        (paper Table 6)

Each variant is a (name, model-config, extra-CLI-flags) triple; results are
written under outputs/<dataset>/main/ablation_<group>_<variant>/seed_<s>/ and
finished runs are skipped so the job can be resumed at any time.

Usage:
    python scripts/run_ablations.py --group shortcuts
    python scripts/run_ablations.py --group imbalance --datasets hdfs --seeds 42 123 256
    python scripts/run_ablations.py --group boundary --datasets hdfs ssh
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


FULL = "configs/model/shortcut_full.yaml"

GROUPS: dict[str, list[tuple[str, str, list[str]]]] = {
    "shortcuts": [
        ("full", FULL, []),
        ("no_entity_embed", "configs/model/shortcut_no_entity_embed.yaml", []),
        ("no_entity_prior", "configs/model/shortcut_no_entity_prior.yaml", []),
        ("no_action_prior", "configs/model/shortcut_no_action_prior.yaml", []),
        ("shuffle_entity", FULL, ["--shuffle-entity-id"]),
        ("entity_unk", FULL, ["--entity-unk"]),
        ("no_action_source", "configs/model/shortcut_no_action_source.yaml", []),
        ("no_status_source", "configs/model/shortcut_no_status_source.yaml", []),
        ("mask_status_words", FULL, ["--mask-status-words"]),
        ("template_time_only", "configs/model/shortcut_template_time.yaml", []),
    ],
    "imbalance": [
        ("bce", FULL, ["--loss-type", "bce"]),
        ("weighted_bce", FULL, ["--loss-type", "weighted_bce"]),
        ("focal", FULL, ["--loss-type", "focal"]),
        ("cb_focal", FULL, ["--loss-type", "cb_focal"]),
        ("no_prototype", "configs/model/imb_no_prototype.yaml", []),
        ("single_prototype", "configs/model/imb_single_prototype.yaml", []),
        ("multi_prototype", "configs/model/imb_multi_prototype.yaml", []),
        (
            "multi_prototype_div",
            "configs/model/imb_multi_prototype.yaml",
            ["--prototype-diversity"],
        ),
    ],
    "boundary": [
        ("full", "configs/model/rq2_full.yaml", []),
        ("detach", "configs/model/rq2_detach.yaml", []),
        ("independent", "configs/model/rq2_independent.yaml", []),
        ("fixed_window", "configs/model/rq2_fixed_window.yaml", []),
        ("random", "configs/model/rq2_random.yaml", []),
        ("hard_action", "configs/model/rq2_hard_action.yaml", []),
        ("hard_entity", "configs/model/rq2_hard_entity.yaml", []),
    ],
    # Candidate main-model configurations derived from the ladder findings
    # (used to decide the final architecture, not for the ablation table).
    "candidates": [
        ("nograph", "configs/model/cand_nograph.yaml", []),
        ("full_reference", "configs/model/sdhtg.yaml", []),
    ],
}

GROUP_DEFAULTS = {
    "shortcuts": (["hdfs", "ssh"], [42, 123, 256, 512, 1024]),
    "imbalance": (["hdfs", "ssh"], [42, 123, 256]),
    "boundary": (["hdfs", "ssh"], [42, 123, 256, 512, 1024]),
    "candidates": (["hdfs", "ssh"], [42, 123, 256, 512, 1024]),
}

# Variants whose resolved model config is identical to the main model, i.e.
# already produced by the ladder (ladder_full/L7) with the same protocol.
# They are the reference rows of the paper tables and are NOT re-run unless
# --include-reference is passed.
REFERENCE_VARIANTS = {
    "shortcuts": {"full"},
    "imbalance": {"cb_focal", "multi_prototype"},
    "boundary": {"full"},
    "candidates": {"full_reference"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument(
        "--include-reference", action="store_true",
        help="also re-run variants identical to the main model "
             "(reference rows already available from ladder_full/L7)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only list the runs that would be executed (skips finished ones)",
    )
    args = parser.parse_args()

    default_datasets, default_seeds = GROUP_DEFAULTS[args.group]
    datasets = args.datasets or default_datasets
    seeds = args.seeds or default_seeds
    variants = GROUPS[args.group]
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in variants if v[0] in wanted]
    skipped = set()
    if not args.include_reference:
        skipped = REFERENCE_VARIANTS.get(args.group, set())
        variants = [v for v in variants if v[0] not in skipped]
        if skipped:
            print(
                "reference variant(s) "
                f"{sorted(skipped)} reuse ladder_full/L7 (identical config, "
                "same protocol); pass --include-reference to re-run them"
            )

    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"
    pending = 0

    for dataset in datasets:
        config_path = Path(f"configs/experiment/{dataset}.yaml")
        if not config_path.is_file():
            print(f"!! missing config {config_path}")
            continue
        output_dir = Path(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))["output_dir"]
        )
        for name, model_config, extra in variants:
            if not Path(model_config).is_file():
                print(f"!! missing model config {model_config} ({name})")
                continue
            tag = f"ablation_{args.group}_{name}"
            for seed in seeds:
                if (output_dir / f"{tag}/seed_{seed}/result.json").is_file():
                    print(f"== {dataset} {name} seed {seed}: cached")
                    continue
                if args.dry_run:
                    pending += 1
                    print(f"== {dataset} {name} seed {seed}: PENDING")
                    continue
                print(f"== {dataset} {name} seed {seed}: running", flush=True)
                code = subprocess.call(
                    [
                        sys.executable, "scripts/train.py",
                        "--config", str(config_path),
                        "--model-config", model_config,
                        "--seed", str(seed),
                        "--tag", tag,
                        "--max-epochs", str(args.max_epochs),
                        "--pretrain-epochs", "0",
                        "--skip-pretrain",
                        "--mask-template-prob", "0.0",
                        *extra,
                    ],
                    env=environment,
                )
                if code != 0:
                    print(f"!! {dataset} {name} seed {seed} failed ({code})")
    if args.dry_run:
        print(f"[dry-run] group {args.group}: {pending} runs pending")
    else:
        print(f"ablation group {args.group} finished")


if __name__ == "__main__":
    main()
