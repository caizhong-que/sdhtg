# -*- coding: utf-8 -*-
"""Pretty-print the efficiency benchmark (paper Table 11)."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("outputs/efficiency_benchmark.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    header = (
        f"{'model':<12}{'T':>5}{'train(ms)':>12}"
        f"{'infer(ms/样本)':>16}{'peak(GB)':>10}{'edges':>10}"
    )
    print(header)
    print("-" * len(header))
    for model, entries in data.items():
        for steps in sorted(entries, key=lambda value: int(value)):
            metrics = entries[steps]
            print(
                f"{model:<12}{int(steps):>5}{metrics['train_ms']:>12.1f}"
                f"{metrics['infer_ms_per_sample']:>16.3f}"
                f"{metrics['peak_memory_gb']:>10.2f}"
                f"{metrics['graph_edges']:>10}"
            )


if __name__ == "__main__":
    main()
