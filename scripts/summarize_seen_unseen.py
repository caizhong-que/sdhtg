# -*- coding: utf-8 -*-
"""summarize_seen_unseen.py -- print the entity-holdout group metrics."""

from __future__ import annotations

import glob
import json
import statistics as st


def main() -> None:
    for dataset in ("bgl", "openstack"):
        rows = []
        for path in sorted(glob.glob(
                f"outputs/{dataset}/main/seen_unseen/entholdout20_L7_seed_*.json")):
            rows.append(json.load(open(path, encoding="utf-8")))
        if not rows:
            print(f"{dataset}: no results")
            continue
        print(f"===== {dataset} (n={len(rows)} seeds) =====")
        for name in ("seen", "holdout", "all"):
            group = [row["groups"].get(name) for row in rows]
            group = [g for g in group if g]
            if not group:
                continue
            samples = st.mean(g["samples"] for g in group)
            positives = st.mean(g["positives"] for g in group)
            auprc = [g["auprc"] for g in group if g["auprc"] is not None]
            f1 = st.mean(g["f1"] for g in group)
            precision = st.mean(g["precision"] for g in group)
            recall = st.mean(g["recall"] for g in group)
            print("  {:<8} n={:<7.0f} pos={:<6.0f} AUPRC={} F1={:.4f} P={:.4f} R={:.4f}".format(
                name, samples, positives,
                f"{st.mean(auprc):.4f}" if auprc else "n/a", f1, precision, recall))
            flags = sorted({bool(g["informative"]) for g in group})
            print(f"           informative={flags}")


if __name__ == "__main__":
    main()
