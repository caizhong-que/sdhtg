# -*- coding: utf-8 -*-
"""
plot_case_figure.py -- section 6.6 case figure (correct / false positive /
false negative) for a finished run.

The figure makes one claim: a sample-level decision can be traced back to
specific events, and the strength of that evidence agrees with the decision.
Each row is one case and carries three panels:

    timeline   action/entity boundary probabilities over the session, with the
               evidence nodes marked at the events that produced them
    subgraph   the top-k anomalous subgraph: node bands per level, node colour
               by anomaly evidence, edge width by message weight
    readout    anomaly probability against the calibrated threshold, the level
               gate pi, and the nearest normal prototype

Run with any Python that has matplotlib, e.g. the base conda environment:

    D:\\anaconda3\\python.exe scripts/plot_case_figure.py \
        --cases outputs/hdfs/main/ladder_full/L7/interpretability/cases_seed42.json \
        --out-dir "E:\\SDHTG\\文章\\初稿\\figure"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
}

NODE_COLOR = {
    "status": PALETTE["blue_secondary"],
    "action": PALETTE["violet"],
    "entity": PALETTE["red_strong"],
}
NODE_BAND = {"entity": 2.0, "action": 1.0, "status": 0.0}
ROW_TITLE = {
    "correct": "correct detection (true positive)",
    "fp": "false positive",
    "fn": "false negative",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def shorten(text: str, width: int = 46) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def draw_timeline(ax, case: dict, show_labels: bool) -> None:
    timeline = case["timeline"]
    events = [row["event"] for row in timeline]
    action = [row["action_boundary"] for row in timeline]
    entity = [row["entity_boundary"] for row in timeline]
    ax.plot(events, action, color=PALETTE["blue_main"], lw=1.0, label="action boundary")
    ax.plot(
        events,
        entity,
        color=PALETTE["teal"],
        lw=1.0,
        ls="--",
        label="entity boundary",
    )

    # Mark the events that produced the selected evidence nodes.
    for node in case["selected_nodes"]:
        colour = NODE_COLOR.get(node["node_type"], PALETTE["neutral_mid"])
        ax.plot(
            node["events"],
            [1.08] * len(node["events"]),
            marker="v",
            ms=3.4,
            ls="none",
            mfc=colour,
            mec="white",
            mew=0.3,
            alpha=0.95,
            zorder=5,
        )
    ax.set_ylim(-0.06, 1.22)
    ax.set_xlim(-1, max(len(events), 1))
    ax.set_yticks([0, 0.5, 1.0])
    if show_labels:
        ax.set_xlabel("event index")
        ax.set_ylabel("boundary\nprobability")
    else:
        ax.set_xticklabels([])
        ax.tick_params(labelbottom=False)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.62), fontsize=6)


def draw_subgraph(ax, case: dict) -> None:
    nodes = {node["node_type"] + f"#{node['local_index']}": node for node in case["selected_nodes"]}
    scores = [node["score"] for node in case["selected_nodes"]] or [0.0]
    limit = max(abs(min(scores)), abs(max(scores)), 1e-3)
    masses = [node["mass"] for node in case["selected_nodes"]] or [1.0]
    max_mass = max(masses)
    weights = [edge["weight"] for edge in case["subgraph_edges"]] or [1.0]
    max_weight = max(weights)

    for edge in case["subgraph_edges"]:
        source = nodes.get(edge["edge_type"][0] + f"#{edge['source']}")
        target = nodes.get(edge["edge_type"][2] + f"#{edge['target']}")
        if source is None or target is None:
            continue
        relation = edge["edge_type"][1]
        ax.plot(
            [source["position"], target["position"]],
            [NODE_BAND[source["node_type"]], NODE_BAND[target["node_type"]]],
            color=PALETTE["neutral_mid"] if relation in ("belongs_to", "contains") else PALETTE["neutral_light"],
            lw=0.4 + 1.6 * (edge["weight"] / max_weight),
            alpha=0.55,
            zorder=1,
        )

    for node in case["selected_nodes"]:
        size = 8.0 + 34.0 * np.log1p(node["mass"]) / np.log1p(max_mass)
        ax.scatter(
            node["position"],
            NODE_BAND[node["node_type"]],
            s=size,
            c=[node["score"]],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            edgecolors=PALETTE["neutral_dark"],
            linewidths=0.4,
            zorder=3,
        )
    for band, label in (("entity", "entity"), ("action", "action"), ("status", "status")):
        ax.text(
            -0.02,
            NODE_BAND[band],
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.5,
            color=PALETTE["neutral_dark"],
        )
    ax.set_yticks([])
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlim(-1, max(case["length"], 1))
    ax.set_xlabel("event position")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)


def draw_readout(ax, case: dict, threshold: float) -> None:
    ax.axis("off")
    probability = case["anomaly_probability"]
    decision = probability >= threshold
    truth = case["label"] == 1
    verdict = "TP" if decision == truth else ("FP" if decision else "FN")
    lines = [
        f"p = {probability:.3f}   τ = {threshold:.3f}",
        f"decision: {'anomaly' if decision else 'normal'}",
        f"truth:    {'anomaly' if truth else 'normal'}  ({verdict})",
        f"prototype P{case['prototype']['nearest']},  d = {case['prototype']['distance']:.2f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=6.2, linespacing=1.5)

    weights = case["level_weights"]
    order = [("status", PALETTE["blue_secondary"]), ("action", PALETTE["violet"]), ("entity", PALETTE["red_strong"])]
    left = 0.0
    for level, colour in order:
        width = float(weights.get(level, 0.0))
        ax.barh(-0.22, width, left=left, height=0.16, color=colour, edgecolor="white", linewidth=0.4)
        if width > 0.12:
            ax.text(left + width / 2, -0.22, level, ha="center", va="center", fontsize=5.6, color="white")
        left += width
    ax.text(0.0, -0.05, "level gate π", ha="left", va="bottom", fontsize=6.4)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.45, 1.05)

    top = max(case["selected_nodes"], key=lambda node: node["score"], default=None)
    if top and top.get("templates"):
        ax.text(
            0.0,
            -0.34,
            "top evidence:\n" + shorten(top["templates"][0], 34),
            ha="left",
            va="top",
            fontsize=5.5,
            color=PALETTE["neutral_dark"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True,
                        help="cases_seed*.json produced by interpret_cases.py")
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_cases")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--order", default="correct,fp,fn")
    args = parser.parse_args()

    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    order = [name.strip() for name in args.order.split(",") if name.strip()]
    cases = [(name, payload["cases"][name]) for name in order if name in payload["cases"]]
    if not cases:
        raise SystemExit("no usable cases in " + args.cases)

    fig = plt.figure(figsize=(7.2, 2.15 * len(cases) + 0.35))
    grid = GridSpec(
        len(cases),
        3,
        figure=fig,
        width_ratios=[3.0, 2.4, 2.0],
        wspace=0.42,
        hspace=0.85,
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.07,
    )
    for row, (name, case) in enumerate(cases):
        last = row == len(cases) - 1
        ax_time = fig.add_subplot(grid[row, 0])
        ax_graph = fig.add_subplot(grid[row, 1])
        ax_read = fig.add_subplot(grid[row, 2])
        draw_timeline(ax_time, case, show_labels=last)
        draw_subgraph(ax_graph, case)
        if not last:
            ax_graph.set_xticklabels([])
            ax_graph.set_xlabel("")
        draw_readout(ax_read, case, threshold)
        ax_time.set_title(
            f"{ROW_TITLE.get(name, name)}  ·  {shorten(case['sample_id'], 34)}",
            loc="left",
            fontsize=7.4,
            pad=4.5,
        )

    handles = [
        Line2D([], [], color=PALETTE["blue_main"], lw=1.0, label="action boundary"),
        Line2D([], [], color=PALETTE["teal"], lw=1.0, ls="--", label="entity boundary"),
    ] + [
        Line2D([], [], marker="v", ls="none", mfc=NODE_COLOR[level], mec="white",
               ms=3.6, label=f"{level} evidence")
        for level in ("status", "action", "entity")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.004),
        fontsize=6.4,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.name
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=args.dpi, bbox_inches="tight")

    csv_path = out_dir / f"{args.name}_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["case", "sample_id", "label", "anomaly_probability", "threshold",
             "pi_status", "pi_action", "pi_entity", "nearest_prototype",
             "prototype_distance", "evidence_nodes", "subgraph_edges"]
        )
        for name, case in cases:
            weights = case["level_weights"]
            writer.writerow(
                [name, case["sample_id"], case["label"],
                 f"{case['anomaly_probability']:.6f}", f"{threshold:.6f}",
                 f"{weights.get('status', 0):.4f}", f"{weights.get('action', 0):.4f}",
                 f"{weights.get('entity', 0):.4f}", case["prototype"]["nearest"],
                 f"{case['prototype']['distance']:.4f}", len(case["selected_nodes"]),
                 len(case["subgraph_edges"])]
            )
    print(f"wrote {stem}.svg / .pdf / .png and {csv_path}")


if __name__ == "__main__":
    main()
