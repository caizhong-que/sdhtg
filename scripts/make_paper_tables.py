# -*- coding: utf-8 -*-
"""
make_paper_tables.py -- render LaTeX rows for Table 5 (main results) and
Table 7 (module ablation) from ladder_full outputs.

For each dataset with a completed ladder, prints:
    - main-model row (L7) with AUPRC/AUROC/F1/P/R  (mean +/- std over seeds)
    - the full L0-L7b ablation table

Usage:
    python scripts/make_paper_tables.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


LEVELS = [f"L{i}" for i in range(8)] + ["L7b"]
LEVEL_LABELS = {
    "L0": "基础 GRU",
    "L1": "+FiLM",
    "L2": "+动作边界/层次聚合",
    "L3": "+嵌套实体边界",
    "L4": "+时间边",
    "L5": "+语义边",
    "L6": "+跨层消息",
    "L7": "+多正常原型",
    "L7b": "+原型多样/均衡",
}


def load_dataset(dataset: str) -> dict[str, list[dict]]:
    root = Path(f"outputs/{dataset}/main/ladder_full")
    if not root.is_dir():
        return {}
    results: dict[str, list[dict]] = {}
    for level in LEVELS:
        level_dir = root / level
        if not level_dir.is_dir():
            continue
        rows = []
        for seed_dir in sorted(level_dir.glob("seed_*")):
            result_path = seed_dir / "result.json"
            if not result_path.is_file():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            test = payload.get("test") or {}
            rows.append(
                {
                    "auprc": test.get("auprc"),
                    "auroc": test.get("auroc"),
                    "f1": test.get("f1"),
                    "precision": test.get("precision"),
                    "recall": test.get("recall"),
                }
            )
        if rows:
            results[level] = rows
    return results


def fmt(values: list[float], digits: int = 4) -> str:
    arr = [v for v in values if v is not None]
    if not arr:
        return "N/A"
    mean = statistics.mean(arr)
    std = statistics.stdev(arr) if len(arr) > 1 else 0.0
    return f"${mean:.{digits}f}\\pm${std:.{digits}f}"


def main() -> None:
    for dataset in ("ssh", "openstack", "bgl", "hdfs", "thunderbird"):
        results = load_dataset(dataset)
        if not results:
            print(f"% {dataset}: ladder not complete, skipped")
            continue
        print(f"% ===== {dataset} =====")
        if "L7" in results:
            main = results["L7"]
            print(
                "MAIN: "
                f"AUPRC {fmt([r['auprc'] for r in main])}  "
                f"AUROC {fmt([r['auroc'] for r in main])}  "
                f"F1 {fmt([r['f1'] for r in main])}  "
                f"P {fmt([r['precision'] for r in main])}  "
                f"R {fmt([r['recall'] for r in main])}"
            )
        print("ABLATION:")
        previous: dict[str, list[float]] | None = None
        for level in LEVELS:
            rows = results.get(level)
            if not rows:
                continue
            auprc = [r["auprc"] for r in rows]
            f1 = [r["f1"] for r in rows]
            delta = ""
            if previous is not None:
                delta = (
                    f"  [ΔAUPRC {statistics.mean(auprc) - statistics.mean(previous['auprc']):+.4f}  "
                    f"ΔF1 {statistics.mean(f1) - statistics.mean(previous['f1']):+.4f}]"
                )
            print(
                f"  {level} {LEVEL_LABELS[level]}: "
                f"AUPRC {fmt(auprc)}  F1 {fmt(f1)}{delta}"
            )
            previous = {"auprc": auprc, "f1": f1}


if __name__ == "__main__":
    main()
