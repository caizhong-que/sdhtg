# -*- coding: utf-8 -*-
"""audit_final_config.py -- check manuscript protocol claims against the code."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


DATASETS = ["bgl", "hdfs", "openstack", "ssh", "thunderbird"]
LEVELS = Path("outputs/_ladder/main/ladder_full")


def split_report(dataset):
    config = yaml.safe_load(Path(f"configs/data/{dataset}.yaml").read_text(encoding="utf-8"))
    seed = config.get("split_seed")
    grouped = bool(config.get("group_by_source_event", False))
    frame = pd.read_parquet(f"data/processed/{dataset}/sessions.parquet",
                            columns=["split", "start_timestamp"])
    stats = frame.groupby("split")["start_timestamp"].agg(["min", "max"])
    ordered = (stats.loc["train", "max"] <= stats.loc["validation", "min"]
               and stats.loc["validation", "max"] <= stats.loc["test", "min"])
    protocol = "temporal" if ordered else "random"
    print("  {:<12} split_seed={:<6} anti_leak_group={:<6} time_ordered={:<6} -> {}"
          .format(dataset, str(seed), str(grouped), str(ordered), protocol))


def flags(level):
    payload = yaml.safe_load((LEVELS / f"model_{level}.yaml").read_text(encoding="utf-8"))
    return payload["model"]["ablation"]


def main():
    print("=== 1. split protocol of each processed cache ===")
    for dataset in DATASETS:
        split_report(dataset)

    print("")
    print("=== 2. ablation flags of the ladder levels ===")
    for level in ("L0", "L2", "L6", "L7", "L7b"):
        enabled = [name for name, value in flags(level).items() if value]
        print("  {:<4} enabled: {}".format(level, ", ".join(enabled) if enabled else "(none)"))

    print("")
    print("=== 3. prototype / hierarchy configuration recorded per run ===")
    sample = Path("outputs/ssh/main/ladder_full/L7/seed_42/cli_overrides.json")
    effective = json.loads(sample.read_text(encoding="utf-8"))["effective_model_config"]
    for key, value in effective.items():
        print("  {:<28} {}".format(key, value))
    model = yaml.safe_load(Path("configs/model/sdhtg.yaml").read_text(encoding="utf-8"))["model"]
    for key in ("num_normal_prototypes", "prototype_temperature",
                "prototype_similarity_threshold", "prototype_margin"):
        print("  sdhtg.yaml {:<16} {}".format(key, model.get(key)))
    print("  sdhtg.yaml hierarchy        {}".format(model.get("hierarchy")))
    loss = yaml.safe_load(Path("configs/experiment/ssh.yaml").read_text(encoding="utf-8"))["loss"]
    for key in ("prototype_weight", "prototype_margin", "prototype_temperature",
                "prototype_diversity_weight", "prototype_balance_weight",
                "effective_number_beta", "focal_gamma"):
        print("  ssh.yaml loss {:<22} {}".format(key, loss.get(key)))


if __name__ == "__main__":
    main()
