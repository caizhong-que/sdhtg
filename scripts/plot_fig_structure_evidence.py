# -*- coding: utf-8 -*-
"""
plot_fig_structure_evidence.py -- Figure 5: structure and evidence verification.

Core conclusion: the learned soft boundary locates a genuine junction almost
exactly (Boundary F1 ~0.98) but only when the junction carries local semantic
change, where a hard prior is equally sufficient; and deleting the highest-
evidence nodes changes the anomaly score far more than deleting random or
low-evidence nodes, monotonically with the deletion fraction.

Panels: (a) Boundary F1 by segmentation strategy and construction;
        (b) Fid(k) for top / low / random deletion;
        (c) delta Fid(top - random) as the deletion fraction grows.

Data: semi-synthetic BGL experiment, reproduced from the manuscript tables 6
and 7 (the raw evaluation logs were not persisted; values are the ones the
manuscript reports, see EXPERIMENT_DESIGN.md sections 7 and 9).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import (PALETTE, apply_style, canvas_qa, legend_overlap_qa,
                            save_pub)

import matplotlib.pyplot as plt


# (strategy label, entity-aligned, same-entity)   -- manuscript table 6
BF1 = [
    ("learned (supervised)", 0.980, 0.130),
    ("hard entity change", 0.985, 0.034),
    ("hard template change", 0.654, 0.539),
    ("hard action change", 0.169, 0.105),
    ("fixed window, $w{=}4$", 0.115, 0.117),
    ("fixed window, $w{=}8$", 0.298, 0.335),
    ("fixed window, $w{=}16$", 0.308, 0.363),
    ("random boundary", 0.188, 0.206),
]

# manuscript table 7: fraction -> (top, low, random)
FID = {
    "1%": (-0.13, -0.74, -0.55),
    "5%": (-0.13, -0.80, -0.90),
    "10%": (-0.22, -0.86, -1.07),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_structure_evidence")
    args = parser.parse_args()

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [1.5, 0.85, 0.85], "wspace": 0.42})
    ax1, ax2, ax3 = axes

    labels = [row[0] for row in BF1]
    aligned = [row[1] for row in BF1]
    same = [row[2] for row in BF1]
    y = range(len(labels))
    height = 0.38
    ax1.barh([i + height / 2 for i in y], aligned, height=height,
             color=PALETTE["blue_main"], label="entity-aligned")
    ax1.barh([i - height / 2 for i in y], same, height=height,
             color=PALETTE["red_soft"], label="same-entity")
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels, fontsize=6.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.12)
    ax1.set_xlabel("Boundary F1 (top-$k$, tolerance $\\pm2$)")
    ax1.set_title("(a) boundary localisation", loc="left", fontsize=7.2)
    handles_a = ax1.get_legend_handles_labels()
    ax1.annotate("", xy=(0.995, 0.0 + height / 2), xytext=(0.0, 0.0 + height / 2),
                 arrowprops=dict(arrowstyle="-", color=PALETTE["neutral_light"], lw=0.6))

    fractions = list(FID)
    xs = range(len(fractions))
    series = [
        ("top-$k$ evidence", 0, PALETTE["red_strong"], "-", "o"),
        ("low-evidence", 1, PALETTE["blue_secondary"], "--", "s"),
        ("random", 2, PALETTE["neutral_mid"], ":", "^"),
    ]
    for label, index, colour, style, marker in series:
        ax2.plot(list(xs), [FID[f][index] for f in fractions], color=colour, ls=style,
                 marker=marker, ms=3.6, lw=1.0, label=label)
    ax2.axhline(0, color=PALETTE["neutral_light"], lw=0.6)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(fractions)
    ax2.set_xlabel("deleted fraction")
    ax2.set_ylabel("$\\mathrm{Fid}(k)$")
    ax2.set_title("(b) faithfulness", loc="left", fontsize=7.2)
    handles_b = ax2.get_legend_handles_labels()

    delta = [FID[f][0] - FID[f][2] for f in fractions]
    ax3.bar(list(xs), delta, color=PALETTE["green_3"], width=0.55,
            edgecolor=PALETTE["neutral_dark"], linewidth=0.5)
    for x, value in zip(xs, delta):
        ax3.text(x, value + 0.02, f"{value:+.2f}", ha="center", va="bottom", fontsize=6.0)
    ax3.set_xticks(list(xs))
    ax3.set_xticklabels(fractions)
    ax3.set_xlabel("deleted fraction")
    ax3.set_ylabel("$\\Delta\\mathrm{Fid}$ (top $-$ random)")
    ax3.set_ylim(0, max(delta) * 1.35)
    ax3.set_title("(c) relative faithfulness", loc="left", fontsize=7.2)

    fig.legend(handles_a[0] + handles_b[0], handles_a[1] + handles_b[1],
               loc="lower center", ncol=5, fontsize=6.0,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.93, bottom=0.22, wspace=0.50)
    offenders = canvas_qa(fig) + legend_overlap_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
