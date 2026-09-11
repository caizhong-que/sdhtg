"""
summarize_ablations.py -- build the paper's ablation tables from results.

Reference rows (full model) are read from ladder_full/L7 instead of being
re-run, because the ablation "full" variants are config-identical to the main
model. Emits human-readable tables and LaTeX rows for

    Table 6  boundary source / cross-level gradient
    Table 8  class-imbalance loss and prototype settings
    Table 9  input-semantics and entity-shortcut ablations

Usage:
    python scripts/summarize_ablations.py
    python scripts/summarize_ablations.py --datasets hdfs ssh
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


GROUPS = {
    "shortcuts": [
        ("完整输入", None),
        ("移除实体嵌入", "no_entity_embed"),
        ("移除实体边界先验", "no_entity_prior"),
        ("移除动作边界先验", "no_action_prior"),
        ("打乱实体ID", "shuffle_entity"),
        ("测试实体全UNK", "entity_unk"),
        ("移除动作源", "no_action_source"),
        ("移除状态源", "no_status_source"),
        ("屏蔽显式异常词", "mask_status_words"),
        ("仅模板与时间", "template_time_only"),
    ],
    "imbalance": [
        ("CB-Focal + 多原型（完整）", None),
        ("BCE", "bce"),
        ("加权BCE", "weighted_bce"),
        ("Focal", "focal"),
        ("无原型", "no_prototype"),
        ("单原型", "single_prototype"),
        ("多原型", "multi_prototype"),
        ("多原型+多样性/均衡", "multi_prototype_div"),
    ],
    "boundary": [
        ("完整SDHTG（学习边界）", None),
        ("跨层边权停止梯度", "detach"),
        ("独立双边界", "independent"),
        ("固定窗口边界", "fixed_window"),
        ("随机边界", "random"),
        ("硬动作变化边界", "hard_action"),
        ("硬实体变化边界", "hard_entity"),
    ],
}


def load(dataset: str, tag: str, subdir: str = "") -> list[tuple[float, float]]:
    root = Path(f"outputs/{dataset}/main/{tag}")
    if subdir:
        root = root / subdir
    if not root.is_dir():
        return []
    values = []
    for seed_dir in sorted(root.glob("seed_*")):
        path = seed_dir / "result.json"
        if path.is_file():
            test = json.loads(path.read_text(encoding="utf-8"))["test"]
            values.append((test["auprc"], test["f1"]))
    return values


def fmt(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f}$\\pm${std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hdfs", "ssh"])
    args = parser.parse_args()

    for group, rows in GROUPS.items():
        for dataset in args.datasets:
            reference = load(dataset, "ladder_full", "L7")
            if not reference and not any(
                load(dataset, f"ablation_{group}_{name}") for _, name in rows if name
            ):
                continue
            print(f"\n===== {group} / {dataset} =====")
            header = f"{'设置':<22}{'AUPRC':>18}{'F1':>18}{'ΔF1':>10}"
            print(header)
            baseline_f1 = statistics.mean([f for _, f in reference]) if reference else None
            for label, variant in rows:
                values = (
                    reference
                    if variant is None
                    else load(dataset, f"ablation_{group}_{variant}")
                )
                if not values:
                    print(f"{label:<22}{'N/A':>18}{'N/A':>18}{'':>10}")
                    continue
                auprc = fmt([a for a, _ in values])
                f1 = fmt([f for _, f in values])
                delta = ""
                if baseline_f1 is not None and variant is not None:
                    delta = f"{statistics.mean([f for _, f in values]) - baseline_f1:+.4f}"
                print(f"{label:<22}{auprc:>18}{f1:>18}{delta:>10}")
                print(
                    f"    \\LaTeX row: {label} & {auprc} & {f1} & {delta or '--'} \\\\"
                )


if __name__ == "__main__":
    main()
