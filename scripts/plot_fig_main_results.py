# -*- coding: utf-8 -*-
"""
plot_fig_main_results.py -- Figure 2: main comparison across five datasets.

Core conclusion: SDHTG wins where the task has dynamic range, and the win lives
in the threshold-calibrated metric rather than in ranking (HDFS: AUPRC slightly
below the best baseline, F1 far above); on the three saturated datasets the
differences are at the 1e-3 level, and on Thunderbird a flat GNN baseline
stays marginally ahead.

Design notes (this is the "direct" version of the comparison):
  (a) the reader should not have to aggregate four methods x two metrics by
      eye, so the left panel shows only the change against the strongest
      baseline per dataset, grouped by metric and annotated with the values;
  (b) absolute F1 is shown as five small multiples, each zoomed to its own
      four-method range, so the saturated datasets stay readable instead of
      collapsing onto one pixel at the ceiling. The caption states that the
      panels use per-dataset ranges.

Data: outputs/<dataset>/main/<tag>/seed_*/result.json (5 seeds, mean +- std).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import PALETTE, apply_style, canvas_qa, legend_overlap_qa, load_results, mean_std, save_pub

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("ssh", "SSH"),
    ("hdfs", "HDFS"),
    ("bgl", "BGL"),
    ("openstack", "OpenStack"),
    ("thunderbird", "Thunderbird"),
]
METHODS = [
    ("TCN", "#484878", "o"),
    ("Transformer", "#7884B4", "^"),
    ("GNN-flat", "#B4C0E4", "s"),
    ("SDHTG", PALETTE["red_strong"], "D"),
]
METRIC_COLOR = {"AUPRC": PALETTE["blue_main"], "F1": PALETTE["red_strong"]}


def collect(output_root: str, metric: str) -> dict[str, dict[str, tuple[float, float]]]:
    table = {}
    for key, label in DATASETS:
        stats = load_results(output_root, key, metric)
        table[label] = {name: mean_std(stats[name]) for name, _, _ in METHODS if name in stats}
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_main_results")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    apply_style()
    f1_table = collect(args.output_root, "f1")
    auprc_table = collect(args.output_root, "auprc")
    labels = [label for _, label in DATASETS]

    fig = plt.figure(figsize=(7.2, 2.85))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.95], wspace=0.30,
                            left=0.085, right=0.99, top=0.87, bottom=0.17)
    ax_delta = fig.add_subplot(grid[0, 0])
    right = grid[0, 1].subgridspec(1, len(labels), wspace=0.42)

    # (a) change against the strongest baseline, per metric
    deltas = {metric: [] for metric in ("AUPRC", "F1")}
    best_names = []
    for label in labels:
        for metric, table in (("AUPRC", auprc_table), ("F1", f1_table)):
            own = table[label]["SDHTG"][0]
            baselines = {k: v[0] for k, v in table[label].items() if k != "SDHTG"}
            best = max(baselines, key=baselines.get)
            deltas[metric].append((own - baselines[best]) * 100)
            if metric == "F1":
                best_names.append(best)
    xs = np.arange(len(labels))
    width = 0.36
    for position, metric in enumerate(("AUPRC", "F1")):
        offset = (position - 0.5) * width
        ax_delta.bar(xs + offset, deltas[metric], width=width,
                     color=METRIC_COLOR[metric], edgecolor=PALETTE["neutral_dark"],
                     linewidth=0.4, label=f"$\\Delta${metric}")
        for x, value in zip(xs + offset, deltas[metric]):
            ax_delta.text(x, value + (0.09 if value >= 0 else -0.09), f"{value:+.2f}",
                          ha="center", va="bottom" if value >= 0 else "top", fontsize=5.6,
                          color=PALETTE["neutral_black"])
    ax_delta.axhline(0, color=PALETTE["neutral_dark"], lw=0.8)
    ax_delta.set_xticks(xs)
    ax_delta.set_xticklabels(labels, fontsize=6.0, rotation=20, ha="right")
    ax_delta.set_ylim(-1.4, 5.2)
    ax_delta.set_ylabel("change vs strongest baseline (pt)")
    ax_delta.set_title("(a) where the gain is", loc="left", fontsize=7.2)
    ax_delta.legend(fontsize=6.0, loc="upper right", handletextpad=0.4)
    for x, name in zip(xs, best_names):
        ax_delta.text(x, -1.25, f"vs {name}", ha="center", va="bottom", fontsize=5.0,
                      color=PALETTE["neutral_mid"], rotation=0)

    # (b) absolute F1 as per-dataset small multiples with local ranges
    for column, label in enumerate(labels):
        ax = fig.add_subplot(right[0, column], sharex=None)
        means = [f1_table[label][name][0] for name, _, _ in METHODS]
        errs = [f1_table[label][name][1] for name, _, _ in METHODS]
        positions = np.arange(len(METHODS))
        for index, (name, colour, marker) in enumerate(METHODS):
            ax.errorbar(index, means[index], yerr=errs[index], marker=marker,
                        ms=3.4 if name != "SDHTG" else 4.2,
                        mfc=colour if name == "SDHTG" else "white",
                        mec=colour, mew=0.9, ls="none", elinewidth=0.7, capsize=1.1,
                        color=colour, zorder=4 if name == "SDHTG" else 3)
        span = max(means) - min(means)
        pad = max(span * 0.28, 0.0015)
        ax.set_ylim(min(means) - pad, max(means) + pad)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_xticks([])
        ax.set_yticks([round(min(means), 3), round(max(means), 3)])
        ax.tick_params(axis="y", labelsize=5.2, length=2)
        ax.set_title(label, fontsize=6.4, pad=3)
        if column == 0:
            ax.set_ylabel("F1 (local range)")
        best_value = max(means[:3])
        ax.axhline(best_value, color=PALETTE["neutral_light"], lw=0.6, ls="--", zorder=1)
    fig.text(0.60, 0.90, "(b) absolute F1, each panel zoomed to its own four-method range",
             ha="center", va="bottom", fontsize=7.2)
    handles = [
        plt.Line2D([], [], marker=marker, ls="none", mfc="white" if name != "SDHTG" else colour,
                   mec=colour, mew=0.9, ms=4.0, label=name)
        for name, colour, marker in METHODS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.0,
               bbox_to_anchor=(0.77, 0.005), handletextpad=0.4, columnspacing=1.2)

    offenders = canvas_qa(fig) + legend_overlap_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
