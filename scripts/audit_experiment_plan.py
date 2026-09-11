# -*- coding: utf-8 -*-
"""
audit_experiment_plan.py -- detect duplicated runs across P0/P1/P2 plans.

A run is identified by (resolved model config signature, effective flags).
The script reports
  (a) duplicates *within* the planned experiments, and
  (b) planned runs already covered by finished ladder results.

Usage:
    python scripts/audit_experiment_plan.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ablations import (  # noqa: E402
    GROUP_DEFAULTS,
    GROUPS,
    REFERENCE_VARIANTS,
)
from sdhtg.models.factory import load_model_config  # noqa: E402


OVERRIDES = {
    "template_vocab_size": 4096,
    "entity_vocab_size": 1024,
    "action_vocab_size": 1024,
    "status_vocab_size": 4096,
}

BASELINE_MODELS = ("tcn", "transformer", "gnn_flat")
BASELINE_DATASETS = ("openstack", "ssh", "bgl", "hdfs", "thunderbird")
BASELINE_SEEDS = (42, 123, 256, 512, 1024)

PRETRAIN_DATASETS = ("ssh",)
PRETRAIN_PROTOCOLS = ("normal_only", "all_train")
PRETRAIN_STRATEGIES = ("random", "hard", "semi_hard", "semantic", "none", "supervised")
PRETRAIN_SEEDS = (42, 123, 256)


def signature(path: str) -> str:
    return repr(load_model_config(path, OVERRIDES))


def collect() -> list[dict]:
    runs: list[dict] = []
    for dataset in BASELINE_DATASETS:
        for model in BASELINE_MODELS:
            for seed in BASELINE_SEEDS:
                runs.append(
                    {
                        "phase": "P0", "dataset": dataset, "seed": seed,
                        "tag": f"baseline_{model}",
                        "model": f"configs/model/{model}.yaml",
                        "flags": (),
                    }
                )
    for group, variants in GROUPS.items():
        datasets, seeds = GROUP_DEFAULTS[group]
        for name, model_config, extra in variants:
            if name in REFERENCE_VARIANTS.get(group, set()):
                continue  # runner skips these: reuse ladder_full/L7
            for dataset in datasets:
                for seed in seeds:
                    runs.append(
                        {
                            "phase": "P1", "dataset": dataset, "seed": seed,
                            "tag": f"ablation_{group}_{name}",
                            "model": model_config,
                            "flags": tuple(extra),
                        }
                    )
    for dataset in PRETRAIN_DATASETS:
        for protocol in PRETRAIN_PROTOCOLS:
            for strategy in PRETRAIN_STRATEGIES:
                for seed in PRETRAIN_SEEDS:
                    runs.append(
                        {
                            "phase": "P2", "dataset": dataset, "seed": seed,
                            "tag": f"pretrain_{protocol}_{strategy}",
                            "model": "configs/model/sdhtg.yaml",
                            "flags": ("pretrain", protocol, strategy),
                        }
                    )
    return runs


def main() -> None:
    runs = collect()
    print(f"planned training runs: {len(runs)}")
    for phase in ("P0", "P1", "P2"):
        print(f"  {phase}: {sum(1 for r in runs if r['phase'] == phase)}")

    # (a) REDUNDANT VARIANT NAMES: >=2 distinct tags sharing one (model, flags)
    #     signature. Same tag across datasets/seeds is expected, not waste.
    buckets: dict[tuple[str, tuple], list[dict]] = defaultdict(list)
    for run in runs:
        buckets[(signature(run["model"]), run["flags"])].append(run)
    print("\n(a) distinct variant names that reduce to the SAME run:")
    redundant_runs = 0
    for (_, flags), entries in buckets.items():
        tags = sorted({e["tag"] for e in entries})
        if len(tags) < 2:
            continue
        keys = {(e["dataset"], e["seed"]) for e in entries}
        extra = sum(
            len({e["tag"] for e in entries if e["dataset"] == ds and e["seed"] == sd}) - 1
            for ds, sd in keys
        )
        redundant_runs += extra
        print(f"  flags={flags or 'default'}: {tags}  -> {extra} redundant runs")
    if not redundant_runs:
        print("  none")

    # (b) flag combinations that do not change the default protocol
    print("\n(b) flags that equal the default protocol (no-op variants):")
    noop = [r for r in runs if r["flags"] == ("--loss-type", "cb_focal")]
    if noop:
        print(
            f"  --loss-type cb_focal: {len(noop)} runs "
            f"(default loss_type is already cb_focal) -> tags "
            f"{sorted({r['tag'] for r in noop})}"
        )
    else:
        print("  none")

    # (c) planned runs already covered by the finished ladder
    covered = {
        "ladder_full/L7 (完整模型)": (signature("configs/model/sdhtg.yaml"), ()),
        "ladder_full/L0 (平铺GRU)": (signature("configs/model/ladder_l0.yaml"), ()),
    }
    print("\n(c) planned runs already covered by the finished ladder:")
    hits = 0
    for name, key in covered.items():
        matched = buckets.get(key, [])
        if matched:
            hits += len(matched)
            print(
                f"  {name}: {len(matched)} runs -> tags "
                f"{sorted({e['tag'] for e in matched})}"
            )
    if not hits:
        print("  none")


if __name__ == "__main__":
    main()
