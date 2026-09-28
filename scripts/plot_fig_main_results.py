# -*- coding: utf-8 -*-
"""
plot_fig_main_results.py -- Figure 2: main comparison across five datasets.

Core conclusion: SDHTG is ahead exactly where the task has dynamic range, and
the advantage lives in the threshold-calibrated metric rather than in ranking;
on the three saturated datasets every method sits at the ceiling, and on
Thunderbird a flat GNN baseline stays marginally ahead.

Design notes (journal experiment figure):
  * two dot plots with one axis per metric, so the rank order of the four
    methods is readable at a glance instead of clustered in a corner;
  * the five datasets are split into "dynamic range" and "saturated" groups by
    a separator, because the saturated group carries no discriminative signal;
  * a diverging heat map gives the per-dataset, per-metric delta against the
    best baseline, which is how the AUPRC/F1 separation becomes explicit.

Data: outputs/<dataset>/main/<tag>/seed_*/result.json (5 seeds, mean +- std).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import (PALETTE, apply_style, canvas_qa, legend_overlap_qa,
                            load_results, mean_std, save_pub)

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
OFFSETS = {"TCN": -0.27, "Transformer": -0.09, "GNN-flat": 0.09, "SDHTG": 0.27}
SPLIT_AFTER = 1  # datasets 0-1 have dynamic range, 2-4 are saturated


def collect(output_root: str, metric: str) -> dict[str, dict[str, tuple[float, float]]]:
    table = {}
    for key, label in DATASETS:
        stats = load_results(output_root, key, metric)
        table[label] = {name: mean_std(stats[name]) for name, _, _ in METHODS if name in stats}
    return table


def draw_dotplot(ax, table, ylabel, title, ylim, annotation=None) -> None:
    labels = [label for _, label in DATASETS]
    xs = np.arange(len(labels))
    for method, colour, marker in METHODS:
        means, errs, positions = [], [], []
        for index, label in enumerate(labels):
            if method not in table[label]:
                continue
            mean, std = table[label][method]
            means.append(mean)
            errs.append(std)
            positions.append(xs[index] + OFFSETS[method])
        ax.errorbar(
            positions,
            means,
            yerr=errs,
            marker=marker,
            ms=3.6 if method != "SDHTG" else 4.4,
            color=colour,
            mfc=colour if method == "SDHTG" else "white",
            mec=colour,
            mew=0.9,
            ls="none",
            elinewidth=0.7,
            capsize=1.1,
            label=method,
            zorder=4 if method == "SDHTG" else 3,
        )
    ax.axvline(SPLIT_AFTER + 0.5, color=PALETTE["neutral_light"], lw=0.7, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=7.2)
    if annotation:
        ax.annotate(annotation[0], xy=annotation[1], xytext=annotation[2],
                    fontsize=5.6, color=PALETTE["neutral_dark"],
                    arrowprops=dict(arrowstyle="-", color=PALETTE["neutral_mid"], lw=0.6))
    ax.text(0.5, 0.985, "dynamic range", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=5.4, color=PALETTE["neutral_mid"])
    ax.text(3.5, 0.985, "saturated", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=5.4, color=PALETTE["neutral_mid"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_main_results")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    apply_style()
    fig = plt.figure(figsize=(7.2, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.85], wspace=0.42,
                            left=0.075, right=0.90, top=0.86, bottom=0.16)
    ax_f1 = fig.add_subplot(grid[0, 0])
    ax_auprc = fig.add_subplot(grid[0, 1])
    ax_delta = fig.add_subplot(grid[0, 2])

    f1_table = collect(args.output_root, "f1")
    auprc_table = collect(args.output_root, "auprc")

    draw_dotplot(ax_f1, f1_table, "F1 (validation-calibrated)", "(a) decision metric",
                 (0.54, 1.012),
                 annotation=("HDFS: $+0.040$ over\nthe best baseline",
                             (1 + OFFSETS["SDHTG"], f1_table["HDFS"]["SDHTG"][0]),
                             (2.15, 0.70)))
    draw_dotplot(ax_auprc, auprc_table, "AUPRC (threshold-free)", "(b) ranking metric",
                 (0.36, 1.012),
                 annotation=("HDFS: Transformer\nstays ahead",
                             (1 + OFFSETS["Transformer"], auprc_table["HDFS"]["Transformer"][0]),
                             (2.2, 0.62)))

    handles = [
        plt.Line2D([], [], marker=marker, ls="none", mfc="white" if name != "SDHTG" else colour,
                   mec=colour, mew=0.9, ms=4.0, label=name)
        for name, colour, marker in METHODS
    ]
    ax_f1.legend(handles=handles, fontsize=5.8, ncol=2, loc="lower left",
                 handletextpad=0.35, columnspacing=0.8, borderpad=0.2)

    # (c) delta against the best baseline per metric
    rows = [label for _, label in DATASETS]
    delta = np.zeros((len(rows), 2))
    best_names = []
    for index, label in enumerate(rows):
        for column, table in enumerate((auprc_table, f1_table)):
            own = table[label]["SDHTG"][0]
            baselines = {k: v[0] for k, v in table[label].items() if k != "SDHTG"}
            best_name = max(baselines, key=baselines.get)
            best_names.append(best_name if column == 1 else None)
            delta[index, column] = (own - baselines[best_name]) * 100
    limit = float(np.abs(delta).max()) * 1.05
    image = ax_delta.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax_delta.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center",
                          fontsize=5.8,
                          color="white" if abs(delta[i, j]) > 0.55 * limit else PALETTE["neutral_black"])
    ax_delta.set_xticks([0, 1])
    ax_delta.set_xticklabels(["$\\Delta$AUPRC", "$\\Delta$F1"], fontsize=6.4)
    ax_delta.set_yticks(range(len(rows)))
    ax_delta.set_yticklabels(rows, fontsize=6.2)
    ax_delta.set_title("(c) vs the best baseline", loc="left", fontsize=7.2)
    ax_delta.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax_delta.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax_delta.grid(which="minor", color="white", lw=0.8)
    ax_delta.tick_params(which="minor", length=0)
    ax_delta.tick_params(axis="y", length=0)
    for spine in ax_delta.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_delta, fraction=0.05, pad=0.06)
    colorbar.ax.tick_params(labelsize=5.6)
    colorbar.set_label("percentage points", fontsize=5.8)

    offenders = canvas_qa(fig) + legend_overlap_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
