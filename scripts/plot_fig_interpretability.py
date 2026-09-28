# -*- coding: utf-8 -*-
"""
plot_fig_interpretability.py -- Figure 4: interpretability evidence.

Core conclusion: the level gate shows which hierarchy level carries the
evidence; the prototype bank is directionally diverse on SSH but collapses onto
a single direction on HDFS while only a few prototypes are used; and the
boundary score ranks action changes far above the base rate even though the
model decides not to place internal boundaries.

Design notes (kept deliberately sparse - one encoding per panel):
  (a) the level gate is a composition (status + action + entity = 1), so it is
      drawn as stacked bars with a single three-entry legend;
  (b) utilisation is drawn as a two-row heat strip, which shows both the idle
      prototypes and the directional collapse without any in-axes text boxes
      (the pairwise-cosine range is printed in the row label);
  (c) prior consistency keeps two curves and two dotted base lines; every
      explanatory note lives in the caption instead of inside the axes.

Data: outputs/<dataset>/main/<tag>/interpretability/{hierarchy,prototypes,boundaries}_seed*.json
(HDFS 3 seeds, SSH 5 seeds) produced by the section 6.6 scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import PALETTE, apply_style, canvas_qa, save_pub

import matplotlib.pyplot as plt
import numpy as np


LEVELS = ("status", "action", "entity")
LEVEL_COLOR = {
    "status": PALETTE["blue_secondary"],
    "action": PALETTE["violet"],
    "entity": PALETTE["red_strong"],
}
DATASETS = (("ssh", "SSH"), ("hdfs", "HDFS"))
DATASET_COLOR = {"SSH": PALETTE["blue_main"], "HDFS": PALETTE["red_strong"]}


def load_hierarchy(root: Path, dataset: str) -> dict:
    rows = []
    for path in sorted(root.glob(
        f"{dataset}/main/ladder_full/L7/interpretability/hierarchy_seed*.csv"
    )):
        with path.open(encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    summary = {}
    for key, label in (("normal", "normal"), ("anomaly", "anomalous")):
        subset = [row for row in rows if (int(row["label"]) == 1) == (key == "anomaly")]
        summary[label] = {
            level: st.fmean(float(row[f"pi_{level}"]) for row in subset)
            for level in LEVELS
        } if subset else {level: 0.0 for level in LEVELS}
    return summary


def load_prototypes(root: Path, dataset: str) -> dict:
    usage, cosine = [], []
    for path in sorted(root.glob(
        f"{dataset}/main/ladder_full/L7/interpretability/prototypes_seed*.json"
    )):
        payload = json.loads(path.read_text(encoding="utf-8"))
        usage.append([entry["usage_share"] for entry in payload["prototypes"]])
        similarity = payload["prototype_similarity"]
        cosine.append((similarity["mean"], similarity["min"], similarity["max"]))
    array = np.asarray(usage)
    return {
        "usage": array.mean(axis=0),
        "cosine_mean": st.fmean(c[0] for c in cosine if c[0] is not None),
        "cosine_min": min(c[1] for c in cosine if c[1] is not None),
        "cosine_max": max(c[2] for c in cosine if c[2] is not None),
    }


def load_boundaries(root: Path, dataset: str) -> dict:
    deciles, base = [], []
    for path in sorted(root.glob(
        f"{dataset}/main/ladder_full/L7/interpretability/boundaries_seed*.json"
    )):
        payload = json.loads(path.read_text(encoding="utf-8"))["action_boundary"]
        deciles.append([row["primary_change_rate"] for row in payload["deciles"]])
        base.append(payload["primary_change_rate"])
    array = np.asarray(deciles, dtype=float)
    return {
        "mean": array.mean(axis=0),
        "std": array.std(axis=0, ddof=1) if array.shape[0] > 1 else np.zeros(array.shape[1]),
        "base": st.fmean(base),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_interpretability")
    args = parser.parse_args()

    root = Path(args.output_root)
    apply_style()
    fig = plt.figure(figsize=(7.2, 2.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.25, 1.05], wspace=0.42,
                            left=0.085, right=0.985, top=0.82, bottom=0.20)
    ax_pi = fig.add_subplot(grid[0, 0])
    ax_usage = fig.add_subplot(grid[0, 1])
    ax_prior = fig.add_subplot(grid[0, 2])

    # (a) stacked composition of the level gate
    entries = [(dataset, label) for dataset, short in DATASETS for label in ("normal", "anomalous")]
    short_names = [f"{short}\n{label}" for (dataset, short) in DATASETS
                   for label in ("normal", "anomalous")]
    bottoms = [0.0] * len(entries)
    xs = np.arange(len(entries))
    for level in LEVELS:
        heights = []
        for dataset, short in DATASETS:
            summary = load_hierarchy(root, dataset)
            for label in ("normal", "anomalous"):
                heights.append(summary[label][level])
        ax_pi.bar(xs, heights, bottom=bottoms, width=0.62,
                  color=LEVEL_COLOR[level], edgecolor="white", linewidth=0.5,
                  label=level)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax_pi.set_xticks(xs)
    ax_pi.set_xticklabels(short_names, fontsize=5.8)
    ax_pi.set_ylim(0, 1.0)
    ax_pi.set_ylabel("level gate $\\pi$ (composition)")
    ax_pi.set_title("(a) evidence by level", loc="left", fontsize=7.2)
    ax_pi.legend(fontsize=5.8, loc="upper center", ncol=3, handletextpad=0.35,
                 columnspacing=0.7, borderpad=0.2, bbox_to_anchor=(0.5, 1.02))

    # (b) prototype utilisation as a heat strip
    usage = np.vstack([load_prototypes(root, dataset)["usage"] for dataset, _ in DATASETS])
    image = ax_usage.imshow(usage, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(usage.shape[0]):
        for column in range(usage.shape[1]):
            value = usage[row, column]
            text = "·" if value < 0.005 else f"{value:.2f}".lstrip("0")
            ax_usage.text(column, row, text, ha="center", va="center", fontsize=5.6,
                          color="white" if value > 0.55 else PALETTE["neutral_dark"])
    row_labels = []
    for dataset, short in DATASETS:
        summary = load_prototypes(root, dataset)
        row_labels.append(
            f"{short}\ncos {summary['cosine_mean']:.2f}\n"
            f"[{summary['cosine_min']:.2f}, {summary['cosine_max']:.2f}]"
        )
    ax_usage.set_yticks(range(len(row_labels)))
    ax_usage.set_yticklabels(row_labels, fontsize=5.8)
    ax_usage.set_xticks(range(usage.shape[1]))
    ax_usage.set_xticklabels([str(i + 1) for i in range(usage.shape[1])], fontsize=5.8)
    ax_usage.set_xlabel("normal prototype index")
    ax_usage.set_title("(b) prototype utilisation", loc="left", fontsize=7.2)
    for spine in ax_usage.spines.values():
        spine.set_visible(False)
    ax_usage.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax_usage, fraction=0.045, pad=0.03)
    colorbar.set_label("usage share", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.6)

    # (c) prior consistency by score decile
    for dataset, short in DATASETS:
        summary = load_boundaries(root, dataset)
        x = np.arange(1, len(summary["mean"]) + 1)
        ax_prior.errorbar(x, summary["mean"], yerr=summary["std"], marker="o", ms=3.0,
                          lw=1.0, color=DATASET_COLOR[short], capsize=1.1,
                          label=short)
        ax_prior.axhline(summary["base"], color=DATASET_COLOR[short], lw=0.7, ls=":")
    ax_prior.set_xticks(range(1, 11))
    ax_prior.set_ylim(0, 1.05)
    ax_prior.set_xlabel("boundary-score decile (1 = highest)")
    ax_prior.set_ylabel("action-change rate")
    ax_prior.set_title("(c) boundary score vs prior", loc="left", fontsize=7.2)
    ax_prior.legend(fontsize=5.8, loc="center right", handletextpad=0.4)

    offenders = canvas_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
