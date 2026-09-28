# -*- coding: utf-8 -*-
"""
plot_fig_efficiency.py -- Figure 7: efficiency and its attribution.

Core conclusion: SDHTG costs one to two orders of magnitude more than flat
baselines, and the cost is host-side graph construction rather than the
quadratic membership matrix; peak memory grows with the batch-size x length
product, so the batch must be chosen jointly with the sequence length.

Panels: (a) training time per step vs T;  (b) inference time per sample vs T;
        (c) peak GPU memory vs T;         (d) where the T=512 step time goes.

Data: outputs/efficiency_benchmark.json (same hardware, warmed up, batch sizes
given in the manuscript table 14). Panel (d) uses the stage attribution from
the sampling profile recorded in EXPERIMENT_DESIGN.md section 10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_common import METHOD_COLOR, PALETTE, apply_style, canvas_qa, save_pub

import matplotlib.pyplot as plt


MODEL_LABEL = {
    "sdhtg": "SDHTG",
    "gnn_flat": "GNN-flat",
    "tcn": "TCN",
    "transformer": "Transformer",
    "gru_flat": "GRU-flat",
}
LENGTHS = ["32", "64", "128", "256", "512"]


def series(benchmark: dict, model: str, key: str) -> list[float]:
    return [benchmark[model][length][key] for length in LENGTHS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="outputs/efficiency_benchmark.json")
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_efficiency")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax_train, ax_infer = axes[0]
    ax_mem, ax_attr = axes[1]
    ts = [int(v) for v in LENGTHS]

    for model, label in MODEL_LABEL.items():
        colour = METHOD_COLOR.get(label, PALETTE["neutral_mid"])
        style = "-" if label in ("SDHTG", "TCN") else "--"
        width = 1.4 if label == "SDHTG" else 1.0
        marker = "o"
        for axis, key in ((ax_train, "train_ms"), (ax_infer, "infer_ms_per_sample"),
                          (ax_mem, "peak_memory_gb")):
            axis.plot(ts, series(benchmark, model, key), color=colour, ls=style,
                      lw=width, marker=marker, ms=3.0, label=label)

    ax_train.set_yscale("log")
    ax_train.set_xscale("log", base=2)
    ax_train.set_xticks(ts)
    ax_train.set_xticklabels([str(v) for v in ts])
    ax_train.set_ylim(15, 3.0e4)
    ax_train.set_xlabel("sequence length $T$")
    ax_train.set_ylabel("training time / step (ms)")
    ax_train.set_title("(a) training cost", loc="left", fontsize=7.2)
    ax_train.legend(fontsize=5.8, ncol=2, handletextpad=0.4, columnspacing=0.8)
    ax_train.annotate("batch 512:\nmemory-limited",
                      (32, series(benchmark, "sdhtg", "train_ms")[0]),
                      textcoords="offset points", xytext=(6, -14), fontsize=5.6,
                      color=PALETTE["neutral_dark"])

    ax_infer.set_yscale("log")
    ax_infer.set_xscale("log", base=2)
    ax_infer.set_xticks(ts)
    ax_infer.set_xticklabels([str(v) for v in ts])
    ax_infer.set_ylim(0.02, 60.0)
    ax_infer.set_xlabel("sequence length $T$")
    ax_infer.set_ylabel("inference time / sample (ms)")
    ax_infer.set_title("(b) inference cost", loc="left", fontsize=7.2)

    ax_mem.set_xscale("log", base=2)
    ax_mem.set_xticks(ts)
    ax_mem.set_xticklabels([str(v) for v in ts])
    ax_mem.axhline(8.0, color=PALETTE["neutral_mid"], lw=0.8, ls=":")
    ax_mem.text(34, 8.35, "8 GB device limit", fontsize=5.6, color=PALETTE["neutral_mid"])
    ax_mem.set_xlabel("sequence length $T$")
    ax_mem.set_ylabel("peak memory (GB)")
    ax_mem.set_title("(c) peak memory", loc="left", fontsize=7.2)
    ax_mem.set_ylim(0, 13.5)

    attribution = [
        ("graph construction", 82.0, PALETTE["red_strong"]),
        ("other host overhead", 15.0, PALETTE["neutral_light"]),
        ("model forward/backward", 3.0, PALETTE["blue_main"]),
    ]
    left = 0.0
    for label, share, colour in attribution:
        ax_attr.barh(0, share, left=left, height=0.45, color=colour,
                     edgecolor="white", linewidth=0.5)
        if share > 8:
            ax_attr.text(left + share / 2, 0, f"{label}\n{share:.0f}%", ha="center",
                         va="center", fontsize=5.9,
                         color="white" if colour != PALETTE["neutral_light"] else PALETTE["neutral_black"])
        else:
            ax_attr.text(left + share + 2, 0.32, f"{label} {share:.0f}%", ha="left",
                         va="center", fontsize=5.6, color=PALETTE["blue_main"])
        left += share
    ax_attr.set_xlim(0, 100)
    ax_attr.set_ylim(-0.6, 0.9)
    ax_attr.set_yticks([])
    ax_attr.set_xlabel("share of step time at $T=512$ (%)")
    ax_attr.set_title("(d) cost attribution", loc="left", fontsize=7.2)

    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    offenders = canvas_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
