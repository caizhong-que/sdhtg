# -*- coding: utf-8 -*-
"""statistics_report.py -- significance families reported in the manuscript.

Section 5.5 promises paired Wilcoxon signed-rank tests with Holm-corrected
q-values and Cliff's delta.  Three families are reported:

    primary   SDHTG (L7) vs the strongest unified-protocol baseline of each
              dataset, one comparison per dataset, F1 and AUPRC
    baseline  SDHTG vs the flat unified-protocol baseline L0, one comparison
              per dataset
    modules   the module steps that the manuscript discusses: +multi-prototype
              (L7 vs L6), the whole ladder (L7 vs L0) and the negative
              diversity/balance step (L7b vs L7) on HDFS and SSH

Holm correction is applied within each family *and metric* (five datasets or
six module steps per family), which is the comparison set each claim relies on.
Results are written to
``outputs/statistics_report.json`` and printed as a LaTeX-ready table body.

Usage:
    python scripts/statistics_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from statistical_tests import cliff_delta, holm_correct, load_per_seed  # noqa: E402

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    raise SystemExit("scipy is required")


DATASETS = ["ssh", "hdfs", "bgl", "openstack", "thunderbird"]
# Unified-protocol baselines, including the flat GRU baseline L0: on SSH the
# GRU baseline is actually stronger than TCN, so "strongest baseline" has to be
# resolved per dataset *and* per metric instead of being hard-coded.
BASELINES = {
    "ladder_full/L0": "GRU-flat (L0)",
    "baseline_tcn": "TCN",
    "baseline_transformer": "Transformer",
    "baseline_gnn_flat": "GNN-flat",
}


def strongest_baseline(dataset: str, metric: str) -> tuple[str, float]:
    root = Path(f"outputs/{dataset}/main")
    best_tag, best_label, best_value = None, None, -np.inf
    for tag, label in BASELINES.items():
        values = load_per_seed(root / tag, metric)
        if not values:
            continue
        mean = float(np.mean(list(values.values())))
        if mean > best_value:
            best_tag, best_label, best_value = tag, label, mean
    return best_tag, best_label


def compare(root: Path, control: str, treatment: str, metric: str):
    control_values = load_per_seed(root / control, metric)
    treatment_values = load_per_seed(root / treatment, metric)
    seeds = sorted(set(control_values) & set(treatment_values))
    if len(seeds) < 2:
        return None
    base = np.asarray([control_values[k] for k in seeds])
    treat = np.asarray([treatment_values[k] for k in seeds])
    if np.allclose(base, treat):
        p_value = 1.0
    else:
        _, p_value = scipy_stats.wilcoxon(treat, base, alternative="two-sided")
    return {
        "seeds": seeds,
        "control_mean": float(base.mean()),
        "treatment_mean": float(treat.mean()),
        "delta": float(treat.mean() - base.mean()),
        "p": float(p_value),
        "cliff": float(cliff_delta(base, treat)),
    }


def main() -> None:
    report: dict[str, dict] = {}

    # ------------------------------------------------------------- family 1/2
    for family in ("primary", "baseline"):
        rows = []
        for dataset in DATASETS:
            root = Path(f"outputs/{dataset}/main")
            for metric, key in (("test_f1", "f1"), ("test_auprc", "auprc")):
                if family == "primary":
                    control, label = strongest_baseline(dataset, metric)
                else:
                    control, label = "ladder_full/L0", "GRU-flat (L0)"
                result = compare(root, control, "ladder_full/L7", metric)
                if result is None:
                    continue
                rows.append({
                    "dataset": dataset,
                    "control": control,
                    "control_label": label,
                    "metric": key,
                    **result,
                })
        for metric_key in ("f1", "auprc"):
            group = [row for row in rows if row["metric"] == metric_key]
            for row, q_value in zip(group, holm_correct([row["p"] for row in group])):
                row["q"] = float(q_value)
        report[family] = rows

    # --------------------------------------------------------------- family 3
    module_rows = []
    for dataset in ("hdfs", "ssh"):
        root = Path(f"outputs/{dataset}/main")
        for control, treatment, label in (
            ("ladder_full/L6", "ladder_full/L7", "多正常原型 L7 vs L6"),
            ("ladder_full/L0", "ladder_full/L7", "完整模型 L7 vs L0"),
            ("ladder_full/L7", "ladder_full/L7b", "原型多样/均衡 L7b vs L7"),
        ):
            result = compare(root, control, treatment, "test_f1")
            if result is None:
                continue
            module_rows.append({
                "dataset": dataset,
                "control": control,
                "treatment": treatment,
                "label": label,
                "metric": "f1",
                **result,
            })
    p_values = [row["p"] for row in module_rows]
    for row, q_value in zip(module_rows, holm_correct(p_values)):
        row["q"] = float(q_value)
    report["modules"] = module_rows

    Path("outputs/statistics_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print("family            dataset/step                      metric    delta      p"
          "        q    cliff")
    for family, rows in report.items():
        for row in rows:
            if family == "primary":
                name = f"vs {row['control_label']}"
            elif family == "baseline":
                name = "vs L0"
            else:
                name = row["label"]
            print("{:<17} {:<12} {:<22} {:<8} {:+.4f} {:>8.4f} {:>8.4f} {:+8.4f}".format(
                family, row["dataset"], name, row["metric"],
                row["delta"], row["p"], row["q"], row["cliff"]))
    print("\nwrote outputs/statistics_report.json")


if __name__ == "__main__":
    main()
