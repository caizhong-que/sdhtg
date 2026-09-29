# -*- coding: utf-8 -*-
"""plot_fig_pr_threshold.py -- figure 4: ranking vs operating point.

Panels:
    (a) SSH precision-recall curves (mean over 5 seeds) for the flat GRU
        baseline, TCN, Transformer, GNN-flat and SDHTG -- the dataset where the
        ranking metric has real dynamic range;
    (b) HDFS F1 against the decision threshold (mean over 2 seeds, 20000-sample
        stratified test subsample), with each method's validation-calibrated
        threshold marked: here AUPRC is saturated for every method, and the
        difference lives in the calibrated operating point.

Data: ``<run>/seed_*/test_scores.csv`` written by ``scripts/dump_test_scores.py``
and the calibrated thresholds in ``<run>/seed_*/result.json``.

Usage:
    python scripts/plot_fig_pr_threshold.py --out-dir figure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import (PALETTE, apply_style, canvas_qa, legend_overlap_qa,
                            save_pub, text_overlap_qa)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


METHODS = [
    ("ladder_full/L0", "GRU-flat", PALETTE["neutral_mid"], "-"),
    ("baseline_tcn", "TCN", PALETTE["blue_secondary"], "-"),
    ("baseline_transformer", "Transformer", PALETTE["teal"], "--"),
    ("baseline_gnn_flat", "GNN-flat", PALETTE["violet"], "-."),
    ("ladder_full/L7", "SDHTG", PALETTE["red_strong"], "-"),
]
RECALL_GRID = np.linspace(0.0, 1.0, 201)
THRESHOLD_GRID = np.linspace(0.0, 1.0, 401)


def load_scores(dataset: str, tag: str):
    pairs = []
    for path in sorted(Path(f"outputs/{dataset}/main/{tag}").glob("seed_*/test_scores.csv")):
        table = np.genfromtxt(path, delimiter=",", names=True)
        pairs.append((table["label"].astype(int), table["score"].astype(float)))
    return pairs


def calibrated_thresholds(dataset: str, tag: str) -> list[float]:
    values = []
    for path in sorted(Path(f"outputs/{dataset}/main/{tag}").glob("seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.append(float(payload["threshold"]["threshold"]))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_pr_threshold")
    args = parser.parse_args()

    apply_style()
    fig, (ax_pr, ax_f1) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---------------------------------------------------------------- panel a
    for tag, label, colour, style in METHODS:
        pairs = load_scores("ssh", tag)
        if not pairs:
            print(f"   !! missing SSH scores for {label}")
            continue
        curves, auprcs = [], []
        for truth, score in pairs:
            precision, recall, _ = precision_recall_curve(truth, score)
            curves.append(np.interp(RECALL_GRID, recall[::-1], precision[::-1]))
            auprcs.append(average_precision_score(truth, score))
        curve = np.mean(curves, axis=0)
        ax_pr.plot(RECALL_GRID, curve, color=colour, lw=1.1, ls=style, label=label)
        anchor = float(np.interp(0.5, RECALL_GRID, curve))
        # SDHTG and GRU-flat sit on top of each other at recall 0.5, so the two
        # labels are pushed apart vertically.
        offset = {"SDHTG": 0.055, "GRU-flat": -0.065}.get(label, 0.028)
        ax_pr.text(0.52, anchor + offset, f"{label} {np.mean(auprcs):.3f}",
                   color=colour, fontsize=5.0, va="bottom", ha="left")
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_pr.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_pr.set_title("(a) SSH precision-recall (mean of 5 seeds)", loc="left", fontsize=7.2)

    # ---------------------------------------------------------------- panel b
    for tag, label, colour, style in METHODS:
        pairs = load_scores("hdfs", tag)
        if not pairs:
            print(f"   !! missing HDFS scores for {label}")
            continue
        curves = []
        for truth, score in pairs:
            f1 = []
            for threshold in THRESHOLD_GRID:
                predicted = score >= threshold
                tp = float(np.sum(predicted & (truth == 1)))
                fp = float(np.sum(predicted & (truth == 0)))
                fn = float(np.sum(~predicted & (truth == 1)))
                f1.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
            curves.append(np.asarray(f1))
        curve = np.mean(curves, axis=0)
        ax_f1.plot(THRESHOLD_GRID, curve, color=colour, lw=1.1, ls=style, label=label)
        thresholds = calibrated_thresholds("hdfs", tag)
        if thresholds:
            centre = float(np.mean(thresholds))
            value = float(np.interp(centre, THRESHOLD_GRID, curve))
            ax_f1.plot(centre, value, marker="o", ms=3.4, mfc="white", mec=colour,
                       mew=0.9, ls="none", zorder=4)
            ax_f1.annotate(f"{value:.3f}", (centre, value), textcoords="offset points",
                           xytext=(2.5, 3.0), fontsize=4.9, color=colour)
    ax_f1.set_xlabel("decision threshold")
    ax_f1.set_ylabel("F1")
    ax_f1.set_xlim(0, 1)
    ax_f1.set_ylim(0, 1.02)
    ax_f1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_f1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_f1.set_title("(b) HDFS F1 vs threshold (mean of 2 seeds)", loc="left", fontsize=7.2)
    ax_f1.text(0.97, 0.045, "circles: validation-calibrated threshold",
               transform=ax_f1.transAxes, ha="right", va="bottom", fontsize=4.9,
               color=PALETTE["neutral_mid"])

    handles = [plt.Line2D([], [], color=colour, lw=1.2, ls=style, label=label)
               for _, label, colour, style in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5.4,
               frameon=False, bbox_to_anchor=(0.5, 0.002), handlelength=1.5,
               columnspacing=1.2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.205, wspace=0.28)
    offenders = canvas_qa(fig) + legend_overlap_qa(fig) + text_overlap_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
