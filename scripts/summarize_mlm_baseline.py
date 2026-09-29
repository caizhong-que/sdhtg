# -*- coding: utf-8 -*-
"""summarize_mlm_baseline.py -- paired comparison for the MLM baseline."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


PLAN = [("ssh", [42, 123, 256, 512, 1024]), ("openstack", [42, 123, 256, 512, 1024]),
        ("bgl", [42])]


def collect(dataset: str, tag: str, metric: str) -> dict[int, float]:
    values = {}
    for path in sorted(Path(f"outputs/{dataset}/main/{tag}").glob("seed_*/result.json")):
        seed = int(path.parent.name.split("_")[1])
        values[seed] = float(json.loads(path.read_text(encoding="utf-8"))["test"][metric])
    return values


def main() -> None:
    print("dataset    method        n   AUPRC        F1           ")
    for dataset, seeds in PLAN:
        for tag, label in (("baseline_transformer", "Transformer"),
                           ("mlm_baseline", "MLM+Transformer"),
                           ("ladder_full/L7", "SDHTG")):
            auprc = collect(dataset, tag, "auprc")
            f1 = collect(dataset, tag, "f1")
            if not auprc or not f1:
                print(f"{dataset:<10} {label:<13} --  (no results yet)")
                continue
            print("{:<10} {:<13} {:<3} {:.4f}±{:.4f}  {:.4f}±{:.4f}".format(
                dataset, label, len(auprc), st.mean(auprc.values()),
                st.stdev(auprc.values()) if len(auprc) > 1 else 0.0,
                st.mean(f1.values()),
                st.stdev(f1.values()) if len(f1) > 1 else 0.0))
        flat = collect(dataset, "baseline_transformer", "auprc")
        mlm = collect(dataset, "mlm_baseline", "auprc")
        common = sorted(set(flat) & set(mlm))
        if len(common) >= 2:
            a = np.asarray([flat[k] for k in common])
            b = np.asarray([mlm[k] for k in common])
            p_value = (1.0 if np.allclose(a, b)
                       else float(scipy_stats.wilcoxon(b, a, alternative="two-sided")[1]))
            print(f"           -> MLM vs Transformer on AUPRC: Δ={b.mean()-a.mean():+.4f} "
                  f"p={p_value:.4f} seeds={common}")
        print()


if __name__ == "__main__":
    main()
