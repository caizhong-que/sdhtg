# -*- coding: utf-8 -*-
"""Summarize label-scarcity runs (outputs/bgl_iter/main/lf_*)."""
import json
import statistics
from pathlib import Path


def main() -> None:
    results: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {}
    for tag in sorted(Path("outputs/bgl_iter/main").glob("lf_*")):
        parts = tag.name.split("_")  # lf_0.01_ladder_l0
        frac, model = parts[1], "_".join(parts[2:4])
        for seed_dir in tag.glob("seed_*"):
            payload = json.loads(
                (seed_dir / "result.json").read_text(encoding="utf-8")
            )
            test = payload["test"]
            results.setdefault((frac, model), {})[seed_dir.name] = (
                payload["best_metric"],
                test["auprc"],
                test["f1"],
            )

    header = f"{'frac':<6}{'model':<12}{'valAUPRC':>10}{'testAUPRC':>12}{'F1':>10}"
    print(header)
    for frac in ("0.01", "0.05", "0.10"):
        for model in ("ladder_l0", "ladder_l7"):
            rows = results.get((frac, model), {})
            if not rows:
                continue
            values = list(rows.values())
            mean = lambda idx: statistics.mean(v[idx] for v in values)
            print(
                f"{frac:<6}{model:<12}{mean(0):>10.4f}{mean(1):>12.4f}{mean(2):>10.4f}"
            )
            for name, value in rows.items():
                print(
                    f"      {name}: val={value[0]:.4f} "
                    f"auprc={value[1]:.4f} f1={value[2]:.4f}"
                )


if __name__ == "__main__":
    main()
