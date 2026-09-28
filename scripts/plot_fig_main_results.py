# -*- coding: utf-8 -*-
"""plot_fig_main_results.py -- Figure 2: main comparison across five datasets.

Core conclusion: on four datasets every unified-protocol method sits at the
ceiling, and the only place with a dynamic range (SSH) is where a flat GRU
baseline is already strong; the gain of SDHTG therefore has to be read in the
threshold-calibrated metric (HDFS: +4.0 pt F1 while AUPRC is flat) rather than
in the ranking metric.

Design (rewritten after review feedback: no more per-dataset zoomed facets and
no dual-unit bars in one panel):
    (a) absolute F1 as a dot plot -- one row per dataset, one marker per method
        (GRU-flat, TCN, Transformer, GNN-flat, SDHTG), with the method spread
        of each row drawn as a light connector and a shaded ceiling band;
    (b) horizontal lollipops of the change against the *strongest unified
        protocol baseline of that metric*, annotated with the value and with
        the name of the baseline it refers to.

Both panels share the dataset order, so the reader moves left-to-right: "where
are we absolute" -> "what is the change and against whom".

Data: outputs/<dataset>/main/<tag>/seed_*/result.json (5 seeds, mean +- std).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import (PALETTE, apply_style, canvas_qa, legend_overlap_qa,
                            save_pub, text_overlap_qa)

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
    ("ladder_full/L0", "GRU-flat", PALETTE["neutral_mid"], "o"),
    ("baseline_tcn", "TCN", PALETTE["blue_secondary"], "v"),
    ("baseline_transformer", "Transformer", PALETTE["teal"], "^"),
    ("baseline_gnn_flat", "GNN-flat", PALETTE["violet"], "s"),
    ("ladder_full/L7", "SDHTG", PALETTE["red_strong"], "D"),
]
METRIC_COLOR = {"auprc": PALETTE["blue_main"], "f1": PALETTE["red_strong"]}
CEILING = 0.995


def collect(output_root: Path, dataset: str, metric: str):
    values = {}
    for tag, name, _, _ in METHODS:
        collected = []
        for seed_dir in sorted((output_root / dataset / "main" / tag).glob("seed_*")):
            result = seed_dir / "result.json"
            if result.is_file():
                payload = json.loads(result.read_text(encoding="utf-8"))["test"]
                if payload.get(metric) is not None:
                    collected.append(float(payload[metric]))
        if collected:
            values[name] = (st.mean(collected), st.stdev(collected), len(collected))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_main_results")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    root = Path(args.output_root)
    f1 = {label: collect(root, key, "f1") for key, label in DATASETS}
    auprc = {label: collect(root, key, "auprc") for key, label in DATASETS}
    labels = [label for _, label in DATASETS]

    apply_style()
    fig = plt.figure(figsize=(7.2, 3.05))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.07,
                            left=0.115, right=0.985, top=0.855, bottom=0.185)
    ax_abs = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1], sharey=ax_abs)

    rows = np.arange(len(labels))
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * 0.205

    # ---------------------------------------------------------------- panel a
    ax_abs.axvspan(CEILING, 1.012, color=PALETTE["neutral_light"], alpha=0.55, zorder=0)
    ax_abs.text(1.0105, rows[-1] + 0.62, "ceiling $\\geq0.995$", fontsize=5.0,
                color=PALETTE["neutral_mid"], ha="right", va="center")
    for row, label in enumerate(labels):
        table = f1[label]
        means = [table[name][0] for _, name, _, _ in METHODS]
        ax_abs.plot([min(means), max(means)], [row, row], color=PALETTE["neutral_light"],
                    lw=1.4, solid_capstyle="round", zorder=1)
        for offset, (_, name, colour, marker) in zip(offsets, METHODS):
            mean, std, _ = table[name]
            filled = name == "SDHTG"
            ax_abs.errorbar(mean, row + offset, xerr=std,
                            marker=marker, ms=2.9 if not filled else 3.8,
                            mfc=colour if filled else "white", mec=colour, mew=0.85,
                            ls="none", elinewidth=0.6, capsize=1.0, color=colour,
                            zorder=4 if filled else 3)
    ax_abs.set_xlim(0.52, 1.012)
    ax_abs.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_abs.set_ylim(rows[-1] + 0.95, -0.75)
    ax_abs.set_yticks(rows)
    ax_abs.set_yticklabels(labels, fontsize=6.0)
    ax_abs.set_xlabel("F1  (mean $\\pm$ std of 5 seeds)", fontsize=6.2)
    ax_abs.set_title("(a) absolute F1", loc="left", fontsize=7.2)
    ax_abs.tick_params(axis="x", labelsize=5.6, length=2.2)
    ax_abs.tick_params(axis="y", length=0)
    for spine in ("left", "right"):
        ax_abs.spines[spine].set_visible(False)

    # ---------------------------------------------------------------- panel b
    for row, label in enumerate(labels):
        for metric, table, offset in (("f1", f1[label], 0.20),
                                      ("auprc", auprc[label], -0.20)):
            own = table["SDHTG"][0]
            baselines = {k: v[0] for k, v in table.items() if k != "SDHTG"}
            best = max(baselines, key=baselines.get)
            delta = (own - baselines[best]) * 100
            colour = METRIC_COLOR[metric]
            ax_delta.barh(row + offset, delta, height=0.20, color=colour,
                          edgecolor=PALETTE["neutral_dark"], linewidth=0.35, zorder=2)
            anchor = delta + (0.10 if delta >= 0 else -0.10)
            ax_delta.text(anchor, row + offset, f"{delta:+.2f} ({best})",
                          fontsize=4.9, color=PALETTE["neutral_black"],
                          ha="left" if delta >= 0 else "right", va="center", zorder=3)
    ax_delta.axvline(0, color=PALETTE["neutral_dark"], lw=0.8, zorder=1)
    ax_delta.set_xlim(-1.6, 5.6)
    ax_delta.set_xticks([0, 1, 2, 3, 4])
    ax_delta.set_xlabel("change vs strongest baseline (pt)", fontsize=6.2)
    ax_delta.set_title("(b) change against the strongest baseline", loc="left",
                       fontsize=7.2)
    ax_delta.tick_params(axis="x", labelsize=5.6, length=2.2)
    ax_delta.tick_params(axis="y", length=0, labelleft=False)
    for spine in ("left", "right"):
        ax_delta.spines[spine].set_visible(False)

    handles = [
        plt.Line2D([], [], marker=marker, ls="none",
                   mfc=colour if name == "SDHTG" else "white", mec=colour,
                   mew=0.85, ms=3.6, label=name)
        for _, name, colour, marker in METHODS
    ] + [
        plt.Rectangle((0, 0), 1, 1, color=METRIC_COLOR["f1"], label="$\\Delta$F1"),
        plt.Rectangle((0, 0), 1, 1, color=METRIC_COLOR["auprc"], label="$\\Delta$AUPRC"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=5.6,
               frameon=False, bbox_to_anchor=(0.55, 0.002), handlelength=1.2,
               columnspacing=1.0, handletextpad=0.35)

    offenders = canvas_qa(fig) + legend_overlap_qa(fig) + text_overlap_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
