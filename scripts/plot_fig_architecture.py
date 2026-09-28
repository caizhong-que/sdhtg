# -*- coding: utf-8 -*-
"""
plot_fig_architecture.py -- Figure 1: SDHTG pipeline overview.

Core conclusion: one outer sample is turned into a sample-level decision by a
chain of five differentiable stages, and the detection loss flows back through
the retained cross-level edge weights into the boundary network.

Run with a Python that has matplotlib (base conda environment):

    D:\\anaconda3\\python.exe scripts/plot_fig_architecture.py \
        --out-dir "E:\\SDHTG\\文章\\初稿\\figure" --name fig_architecture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import PALETTE, apply_style, canvas_qa, save_pub

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


BOX_W, BOX_H = 2.15, 1.42
XS = (0.15, 2.55, 4.95, 7.35)
Y_TOP, Y_BOTTOM = 3.92, 1.28

STAGES = [
    (
        1,
        "Outer sample",
        ["$X_i$: events $e_t$ with", "template, entity, action,", "status and $\\Delta t$",
         "$T\\leq 512$"],
    ),
    (
        2,
        "Causal encoder",
        ["field embeddings $\\rightarrow$ gated", "fusion $\\rightarrow$ causal GRU",
         "$h_t$ depends on $\\leq t$ only"],
    ),
    (
        3,
        "Strategy FiLM",
        ["$(\\gamma_t,\\beta_t)$ from $\\Delta t$ and", "change flags",
         "$h'_t=\\gamma_t\\odot h_t+\\beta_t$"],
    ),
    (
        4,
        "Nested soft boundaries",
        ["action $p^A_t$ and conditional", "entity $q_t$, so",
         "$p^E_t=p^A_tq_t\\leq p^A_t$"],
    ),
    (
        5,
        "Differentiable hierarchy",
        ["soft membership $M$:", "event $\\rightarrow$ action $\\rightarrow$ entity",
         "dense triangular, $O(T^2)$"],
    ),
    (
        6,
        "Heterogeneous graph",
        ["4 node types; temporal,", "semantic and containment",
         "edges carry gradients"],
    ),
    (
        7,
        "Level detection",
        ["$\\pi$-gated level logits +", "soft-min prototype",
         "distance (multi-prototype)"],
    ),
    (
        8,
        "Decision and evidence",
        ["$p(\\mathrm{anomaly})$ vs calibrated", "$\\tau$; top-$k$ subgraph,",
         "$\\pi$ and nearest prototype"],
    ),
]


def draw_box(ax, x, y, index, title, lines, accent) -> None:
    box = FancyBboxPatch(
        (x, y),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.9,
        edgecolor=accent,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.12,
        y + BOX_H - 0.20,
        f"{index}. {title}",
        ha="left",
        va="center",
        fontsize=6.8,
        color=accent,
        zorder=3,
    )
    ax.text(
        x + 0.12,
        y + BOX_H - 0.50,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=5.9,
        color=PALETTE["neutral_dark"],
        linespacing=1.42,
        zorder=3,
    )


def arrow(ax, start, end, colour, style="-|>", width=0.9, dashed=False, rad=0.0, zorder=4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=width,
        color=colour,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(patch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_architecture")
    args = parser.parse_args()

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(-0.05, 10.75)
    ax.set_ylim(0.35, 5.75)
    ax.axis("off")

    accents = [PALETTE["neutral_dark"], PALETTE["blue_main"], PALETTE["blue_secondary"],
               PALETTE["teal"], PALETTE["violet"], PALETTE["violet"],
               PALETTE["red_strong"], PALETTE["red_strong"]]
    for position, (index, title, lines) in enumerate(STAGES):
        row, column = divmod(position, 4)
        x = XS[column]
        y = Y_TOP if row == 0 else Y_BOTTOM
        draw_box(ax, x, y, index, title, lines, accents[position])

    # left-to-right flow inside each row
    for row, y in ((0, Y_TOP), (1, Y_BOTTOM)):
        for column in range(3):
            arrow(
                ax,
                (XS[column] + BOX_W, y + BOX_H / 2),
                (XS[column + 1], y + BOX_H / 2),
                PALETTE["neutral_dark"],
            )

    # wrap-around: stage 4 -> stage 5, routed through the inter-row band
    band_y = 3.42
    arrow(ax, (XS[3] + BOX_W / 2, Y_TOP), (XS[3] + BOX_W / 2, band_y), PALETTE["neutral_dark"])
    arrow(ax, (XS[3] + BOX_W / 2, band_y), (XS[0] + BOX_W / 2, band_y), PALETTE["neutral_dark"],
          style="-", width=0.9)
    arrow(ax, (XS[0] + BOX_W / 2, band_y), (XS[0] + BOX_W / 2, Y_BOTTOM + BOX_H),
          PALETTE["neutral_dark"])
    ax.text(
        (XS[0] + XS[3] + BOX_W) / 2,
        band_y + 0.10,
        "differentiable stages 5–8",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=PALETTE["neutral_mid"],
    )

    # gradient route: detection loss back to the boundary network
    route_x = XS[3] + BOX_W + 0.22
    arrow(ax, (route_x, Y_BOTTOM + BOX_H / 2), (route_x, Y_TOP + BOX_H / 2),
          PALETTE["red_strong"], dashed=True, width=1.0)
    ax.text(
        route_x + 0.14,
        (Y_BOTTOM + Y_TOP + BOX_H) / 2,
        "gradient route",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        color=PALETTE["red_strong"],
    )

    # note on what the two rows produce
    ax.text(
        0.15,
        5.52,
        "one outer sample in, one calibrated decision out; every stage is differentiable",
        ha="left",
        va="center",
        fontsize=6.4,
        color=PALETTE["neutral_black"],
    )

    offenders = canvas_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
