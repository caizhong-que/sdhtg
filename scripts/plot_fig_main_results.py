# -*- coding: utf-8 -*-
"""
plot_fig_main_results.py -- Figure 3: main comparison across five datasets.

Core conclusion: SDHTG leads where the task has dynamic range, and its gain
shows up in the threshold-calibrated metric (HDFS: AUPRC slightly below the
Transformer baseline while F1 is far above it); on saturated datasets it does
not degrade, and on Thunderbird a flat GNN baseline stays slightly ahead.

Panels: (a) AUPRC-F1 plane with one marker per (dataset, method);
        (b) delta F1 of SDHTG against the strongest baseline per dataset.

Data: outputs/<dataset>/main/<tag>/seed_*/result.json (5 seeds, mean +- std).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import (
    METHOD_COLOR,
    PALETTE,
    apply_style,
    canvas_qa,
    load_results,
    mean_std,
    save_pub,
)

import matplotlib.pyplot as plt


DATASETS = [
    ("ssh", "SSH"),
    ("hdfs", "HDFS"),
    ("bgl", "BGL"),
    ("openstack", "OpenStack"),
    ("thunderbird", "Thunderbird"),
]
DATASET_MARKER = {
    "SSH": "o",
    "HDFS": "s",
    "BGL": "^",
    "OpenStack": "D",
    "Thunderbird": "v",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_main_results")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28})
    ax, ax2 = axes

    summary = {}
    for key, label in DATASETS:
        auprc = load_results(args.output_root, key, "auprc")
        f1 = load_results(args.output_root, key, "f1")
        summary[label] = {}
        for method in METHOD_COLOR:
            if method not in auprc or method not in f1:
                continue
            a_mean, a_std = mean_std(auprc[method])
            f_mean, f_std = mean_std(f1[method])
            summary[label][method] = (a_mean, a_std, f_mean, f_std)
            ax.errorbar(
                a_mean,
                f_mean,
                xerr=a_std,
                yerr=f_std,
                marker=DATASET_MARKER[label],
                ms=4.2,
                mfc=METHOD_COLOR[method] if method == "SDHTG" else "white",
                mec=METHOD_COLOR[method],
                mew=1.0,
                ls="none",
                ecolor=METHOD_COLOR[method],
                elinewidth=0.7,
                capsize=1.5,
                alpha=0.95,
                zorder=3,
            )

    for label, points in summary.items():
        for method, (a_mean, _, f_mean, _) in points.items():
            if method != "SDHTG":
                continue
            ax.annotate(
                label,
                (a_mean, f_mean),
                textcoords="offset points",
                xytext=(6, -3 if label in ("SSH", "HDFS") else 4),
                fontsize=6.2,
                color=PALETTE["red_strong"],
            )
    ax.axhline(1.0, color=PALETTE["neutral_light"], lw=0.6, zorder=1)
    ax.axvline(1.0, color=PALETTE["neutral_light"], lw=0.6, zorder=1)
    ax.set_xlabel("AUPRC (threshold-free)")
    ax.set_ylabel("F1 (validation-calibrated)")
    ax.set_title("(a) ranking vs decision metric", loc="left", fontsize=7.2)
    handles = [
        plt.Line2D([], [], marker="o", ls="none", mfc=colour, mec=colour, ms=4.6, label=method)
        for method, colour in METHOD_COLOR.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.2, handletextpad=0.4)

    labels = [label for _, label in DATASETS]
    deltas = []
    for label in labels:
        points = summary[label]
        baselines = {k: v for k, v in points.items() if k != "SDHTG"}
        best = max(baselines.items(), key=lambda kv: kv[1][2])
        deltas.append((label, points["SDHTG"][2] - best[1][2], best[0]))
    xs = range(len(labels))
    colours = [PALETTE["red_strong"] if d >= 0 else PALETTE["blue_secondary"] for _, d, _ in deltas]
    ax2.bar(list(xs), [d for _, d, _ in deltas], color=colours, width=0.6)
    ax2.axhline(0, color=PALETTE["neutral_dark"], lw=0.8)
    for x, (_, d, best) in zip(xs, deltas):
        ax2.text(x, d + (0.004 if d >= 0 else -0.004), f"{d:+.3f}", ha="center",
                 va="bottom" if d >= 0 else "top", fontsize=6.0)
        ax2.text(x, -0.030, f"vs {best}", ha="center", va="top", fontsize=5.4,
                 color=PALETTE["neutral_mid"])
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels, fontsize=6.4)
    ax2.set_ylabel("$\\Delta$F1 vs strongest baseline")
    ax2.set_ylim(-0.055, 0.065)
    ax2.set_title("(b) gain over the strongest baseline", loc="left", fontsize=7.2)

    fig.tight_layout(w_pad=1.8, h_pad=1.2)
    offenders = canvas_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
