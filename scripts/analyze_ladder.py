# -*- coding: utf-8 -*-
"""
analyze_ladder.py -- per-module effectiveness from ladder results.

Reads seed_*/result.json under an experiment tag, computes mean +/- std of
test AUPRC / F1 per level and the delta vs the previous level, so each module
can be admitted or rejected on evidence.

Usage:
    python scripts/analyze_ladder.py --root outputs/bgl_iter/main/ladder_final \
        --metric test_auprc --metric test_f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEVEL_ORDER = [f"L{i}" for i in range(9)] + ["L7b"]


def load_results(root: Path) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for level_dir in sorted(root.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("L"):
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
                    "seed": seed_dir.name,
                    "val_auprc": payload.get("best_metric"),
                    "test_auprc": test.get("auprc"),
                    "test_f1": test.get("f1"),
                    "test_precision": test.get("precision"),
                    "test_recall": test.get("recall"),
                    "epochs": payload.get("epochs"),
                }
            )
        if rows:
            results[level_dir.name] = rows
    return results


def fmt(values: list[float]) -> str:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return "N/A"
    if arr.size == 1:
        return f"{arr[0]:.5f}"
    return f"{arr.mean():.5f}±{arr.std(ddof=1):.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--metrics", nargs="+", default=["test_auprc", "test_f1"])
    args = parser.parse_args()

    results = load_results(Path(args.root))
    ordered = [level for level in LEVEL_ORDER if level in results]
    header = f"{'level':<6}" + "".join(f"{m:>18}" for m in args.metrics)
    print(header)
    previous: dict[str, list[float]] | None = None
    for level in ordered:
        rows = results[level]
        line = f"{level:<6}"
        deltas = []
        for metric in args.metrics:
            values = [r[metric] for r in rows]
            line += f"{fmt(values):>18}"
            if previous is not None and previous.get(metric):
                delta = float(np.mean(values)) - float(np.mean(previous[metric]))
                deltas.append(f"{metric}Δ{delta:+.5f}")
            else:
                deltas.append("")
        print(line)
        if previous is not None:
            print("        " + "  ".join(d for d in deltas if d))
        previous = {metric: [r[metric] for r in rows] for metric in args.metrics}

    print("\nper-seed detail:")
    for level in ordered:
        for row in results[level]:
            print(
                f"  {level} {row['seed']}: val={row['val_auprc']:.5f} "
                f"test_auprc={row['test_auprc']:.5f} f1={row['test_f1']:.5f} "
                f"p={row['test_precision']:.4f} r={row['test_recall']:.4f} "
                f"epochs={row['epochs']}"
            )


if __name__ == "__main__":
    main()
