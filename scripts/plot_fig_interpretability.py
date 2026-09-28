# -*- coding: utf-8 -*-
"""
plot_fig_interpretability.py -- Figure 6: interpretability evidence.

Core conclusion: the level gate pi shows which hierarchy level carries the
anomaly evidence; the prototype bank is directionally diverse on SSH but
collapses to a single direction on HDFS while only a few prototypes are used;
and the boundary score ranks action changes well above the base rate even
though the model decides not to place internal boundaries.

Panels: (a) level gate pi for normal vs anomalous samples (SSH and HDFS);
        (b) prototype usage per prototype with directional-similarity annotation;
        (c) prior-consistency of the boundary score by score decile.

Data: outputs/<dataset>/main/<tag>/interpretability/{hierarchy,prototypes,boundaries}_seed*.json
(HDFS 3 seeds, SSH 5 seeds) produced by the section 6.6 scripts.
"""

from __future__ import annotations

import argparse
import csv
import glob
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


def load_hierarchy(root: Path, dataset: str) -> dict:
    rows = []
    for path in sorted(root.glob(f"{dataset}/main/ladder_full/L7/interpretability/hierarchy_seed*.csv")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    buckets = {"normal": {level: [] for level in LEVELS},
               "anomaly": {level: [] for level in LEVELS}}
    for row in rows:
        key = "anomaly" if int(row["label"]) == 1 else "normal"
        for level in LEVELS:
            buckets[key][level].append(float(row[f"pi_{level}"]))
    summary = {}
    for key, levels in buckets.items():
        summary[key] = {
            level: (st.fmean(values), st.stdev(values) if len(values) > 1 else 0.0)
            for level, values in levels.items()
            if values
        }
    return summary


def load_prototypes(root: Path, dataset: str) -> dict:
    usage, cosine = [], []
    for path in sorted(root.glob(f"{dataset}/main/ladder_full/L7/interpretability/prototypes_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        usage.append([entry["usage_share"] for entry in payload["prototypes"]])
        similarity = payload["prototype_similarity"]
        cosine.append((similarity["mean"], similarity["min"], similarity["max"]))
    if not usage:
        return {}
    array = np.asarray(usage)
    return {
        "usage_mean": array.mean(axis=0),
        "usage_std": array.std(axis=0, ddof=1) if array.shape[0] > 1 else np.zeros(array.shape[1]),
        "cosine_mean": st.fmean(c[0] for c in cosine if c[0] is not None),
        "cosine_min": min(c[1] for c in cosine if c[1] is not None),
        "cosine_max": max(c[2] for c in cosine if c[2] is not None),
    }


def load_boundaries(root: Path, dataset: str) -> dict:
    deciles, base, positive = [], [], []
    for path in sorted(root.glob(f"{dataset}/main/ladder_full/L7/interpretability/boundaries_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))["action_boundary"]
        deciles.append([row["primary_change_rate"] for row in payload["deciles"]])
        base.append(payload["primary_change_rate"])
        positive.append(payload["fraction_score_positive"])
    if not deciles:
        return {}
    array = np.asarray(deciles, dtype=float)
    return {
        "mean": array.mean(axis=0),
        "std": array.std(axis=0, ddof=1) if array.shape[0] > 1 else np.zeros(array.shape[1]),
        "base": st.fmean(base),
        "positive_share": st.fmean(positive),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_interpretability")
    args = parser.parse_args()

    root = Path(args.output_root)
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.0], "wspace": 0.45})
    ax_pi, ax_proto, ax_prior = axes

    # (a) level gate pi, normal vs anomaly
    hierarchy = {dataset: load_hierarchy(root, dataset) for dataset, _ in DATASETS}
    width = 0.13
    group_x = np.arange(len(DATASETS))
    for level_index, level in enumerate(LEVELS):
        for class_index, (key, alpha) in enumerate((("normal", 0.45), ("anomaly", 1.0))):
            offset = (level_index * 2 + class_index - 2.5) * width
            means = [hierarchy[dataset][key].get(level, (0, 0))[0] for dataset, _ in DATASETS]
            stds = [hierarchy[dataset][key].get(level, (0, 0))[1] for dataset, _ in DATASETS]
            ax_pi.bar(group_x + offset, means, width=width, yerr=stds, capsize=1.2,
                      color=LEVEL_COLOR[level], alpha=alpha,
                      edgecolor=PALETTE["neutral_dark"], linewidth=0.4,
                      label=f"{level} ({'anom.' if key == 'anomaly' else 'norm.'})")
    ax_pi.set_xticks(group_x)
    ax_pi.set_xticklabels([label for _, label in DATASETS])
    ax_pi.set_ylabel("level gate $\\pi$")
    ax_pi.set_title("(a) which level carries the evidence", loc="left", fontsize=7.2)
    ax_pi.legend(fontsize=5.4, ncol=2, loc="upper center", handletextpad=0.35,
                 columnspacing=0.7, borderpad=0.2)

    # (b) prototype usage + directional similarity
    summaries = {dataset: load_prototypes(root, dataset) for dataset, _ in DATASETS}
    indices = np.arange(1, 9)
    bar_width = 0.38
    for position, (dataset, label) in enumerate(DATASETS):
        summary = summaries[dataset]
        offset = (position - 0.5) * bar_width
        ax_proto.bar(indices + offset, summary["usage_mean"], width=bar_width,
                     yerr=summary["usage_std"], capsize=1.2,
                     color=PALETTE["red_strong"] if label == "HDFS" else PALETTE["blue_main"],
                     edgecolor=PALETTE["neutral_dark"], linewidth=0.4, label=label)
        ax_proto.text(
            0.02, 0.97 - 0.16 * position,
            f"{label}: cosine {summary['cosine_mean']:.2f} "
            f"[{summary['cosine_min']:.2f}, {summary['cosine_max']:.2f}]",
            transform=ax_proto.transAxes, fontsize=5.6, va="top", ha="left",
            color=PALETTE["neutral_dark"],
        )
    ax_proto.set_xticks(indices)
    ax_proto.set_xlabel("prototype index")
    ax_proto.set_ylabel("usage share")
    ax_proto.set_ylim(0, 1.12)
    ax_proto.set_title("(b) prototype utilisation and collapse", loc="left", fontsize=7.2)
    ax_proto.legend(fontsize=5.8, loc="upper right", handletextpad=0.4)
    ax_proto.text(0.02, 0.36,
                  "HDFS: directions collapse\n(cosine $\\equiv$ 1.00)",
                  transform=ax_proto.transAxes, fontsize=5.6, va="top",
                  color=PALETTE["red_strong"])

    # (c) prior consistency by score decile
    for dataset, label in DATASETS:
        summary = load_boundaries(root, dataset)
        if not summary:
            continue
        x = np.arange(1, len(summary["mean"]) + 1)
        colour = PALETTE["red_strong"] if label == "HDFS" else PALETTE["blue_main"]
        ax_prior.errorbar(x, summary["mean"], yerr=summary["std"], marker="o", ms=3.2,
                          lw=1.0, color=colour, capsize=1.2, label=label)
        ax_prior.axhline(summary["base"], color=colour, lw=0.7, ls=":")
    ax_prior.set_xticks(range(1, 11))
    ax_prior.set_xlabel("boundary-score decile (1 = highest)")
    ax_prior.set_ylabel("action-change rate")
    ax_prior.set_ylim(0, 1.05)
    ax_prior.set_title("(c) boundary score vs prior", loc="left", fontsize=7.2)
    ax_prior.legend(fontsize=5.8, loc="center right", handletextpad=0.4)
    ax_prior.text(0.03, 0.30,
                  "dotted: base rate\ninternal boundary share $=0$",
                  transform=ax_prior.transAxes, fontsize=5.5, va="top",
                  color=PALETTE["neutral_dark"])

    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.19, wspace=0.52)
    offenders = canvas_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
