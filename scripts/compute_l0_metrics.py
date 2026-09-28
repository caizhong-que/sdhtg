# -*- coding: utf-8 -*-
"""compute_l0_metrics.py -- fill MCC / balanced accuracy for the GRU-flat row.

The L0 (flat GRU) runs stored AUPRC/AUROC/precision/recall/F1 but not MCC and
balanced accuracy, which the main-results table also reports.  Both can be
recovered exactly from precision, recall and the test-set composition:

    TP = recall * P,  FN = P - TP,  FP = TP / precision - TP,  TN = N - FP

The script first validates that identity against the rows whose MCC/balanced
accuracy are already in the table (TCN on every dataset); only if the identity
reproduces those values does it emit the L0 numbers.

Usage:
    python scripts/compute_l0_metrics.py
"""

from __future__ import annotations

import io
import json
import math
import re
import statistics as st
from pathlib import Path

import pandas as pd


DATASETS = ["bgl", "hdfs", "openstack", "ssh", "thunderbird"]
VARIANT = "ladder_full/L0"


def test_composition(dataset: str) -> tuple[int, int]:
    frame = pd.read_parquet(f"data/processed/{dataset}/sessions.parquet",
                            columns=["split", "label"])
    test = frame[frame.split == "test"]
    positives = int(test.label.sum())
    return positives, int(len(test) - positives)


def derive(precision: float, recall: float, positives: int, negatives: int):
    tp = recall * positives
    fn = positives - tp
    fp = (tp / precision - tp) if precision > 0 else float("nan")
    tn = negatives - fp
    if min(tp, fp, fn, tn) < 0:
        return None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    return mcc, 0.5 * (tpr + tnr)


def table_values() -> dict[tuple[str, str], tuple[float, float]]:
    """MCC / balanced accuracy already printed in the manuscript table."""
    text = io.open(r"E:\SDHTG\文章\初稿\【1】Manuscript.tex", encoding="utf-8").read()
    block = text[text.index("\\label{tab:main-results}"):]
    block = block[:block.index("\\end{tabular}")]
    values = {}
    dataset = None
    for line in block.split("\n"):
        match = re.search(r"multirow\{4\}\{\*\}\{([A-Za-z]+)\}", line)
        if match:
            dataset = match.group(1).lower()
        row = re.match(r"\s*& (TCN|Transformer|GNN-flat|SDHTG) &(.*)", line)
        if row and dataset:
            cells = [cell.strip() for cell in row.group(2).split("&")]
            if len(cells) >= 7:
                def value(cell: str) -> float:
                    return float(re.sub(r"\\pm.*", "", cell.replace("$", "")).strip())
                values[(dataset, row.group(1))] = (value(cells[5]), value(cells[6]))
    return values


def main() -> None:
    reference = table_values()
    print("validating the derivation against the published TCN rows:")
    for dataset in DATASETS:
        positives, negatives = test_composition(dataset)
        runs = sorted(Path(f"outputs/{dataset}/main/baseline_tcn").glob("seed_*/result.json"))
        if not runs:
            continue
        precisions = [json.loads(p.read_text(encoding="utf-8"))["test"]["precision"] for p in runs]
        recalls = [json.loads(p.read_text(encoding="utf-8"))["test"]["recall"] for p in runs]
        derived = [derive(p, r, positives, negatives) for p, r in zip(precisions, recalls)]
        derived = [d for d in derived if d]
        mcc = st.mean(d[0] for d in derived)
        balanced = st.mean(d[1] for d in derived)
        published = reference.get((dataset, "TCN"))
        print(f"  {dataset:<12} derived MCC={mcc:.4f} BA={balanced:.4f} | "
              f"table MCC={published[0]:.4f} BA={published[1]:.4f}"
              if published else f"  {dataset}: no published row")

    print("\nGRU-flat (L0) rows to insert:")
    for dataset in DATASETS:
        positives, negatives = test_composition(dataset)
        runs = sorted(Path(f"outputs/{dataset}/main/{VARIANT}").glob("seed_*/result.json"))
        payloads = [json.loads(p.read_text(encoding="utf-8"))["test"] for p in runs]
        derived = [derive(p["precision"], p["recall"], positives, negatives) for p in payloads]
        derived = [d for d in derived if d]
        def mean_std(key):
            values = [p[key] for p in payloads]
            return st.mean(values), st.stdev(values)
        auprc, auroc = mean_std("auprc"), mean_std("auroc")
        precision, recall, f1 = mean_std("precision"), mean_std("recall"), mean_std("f1")
        mcc = (st.mean(d[0] for d in derived), st.stdev([d[0] for d in derived]))
        balanced = (st.mean(d[1] for d in derived), st.stdev([d[1] for d in derived]))
        def fmt(pair):
            return f"{pair[0]:.4f}\\pm{pair[1]:.4f}"
        print(f"& GRU-flat & ${fmt(auprc)}$ & ${fmt(auroc)}$ & ${fmt(precision)}$ & "
              f"${fmt(recall)}$ & ${fmt(f1)}$ & ${fmt(mcc)}$ & ${fmt(balanced)}$ \\\\"
              f"   % {dataset} n={len(payloads)}")


if __name__ == "__main__":
    main()
