# -*- coding: utf-8 -*-
"""
plot_fig_sensitivity_robustness.py -- Figure 7: hyper-parameter sensitivity and
parsing-noise robustness.

Core conclusion: the default configuration sits inside the stable region - on
the declared selection criterion (validation AUPRC) no alternative is clearly
better - while several settings that look better on the test metric are worse
on validation, so they are reported rather than adopted; and template-level
parser noise leaves the ranking untouched while the threshold-calibrated F1
moves in both directions, i.e. the F1 movement is calibration noise rather
than a systematic loss of detection ability.

Panels: (a) per-variant change of validation AUPRC and test F1 against the
        default model;  (b) paired change (corrupted test minus clean test of
        the *same* weights and the *same* threshold) under four parsing-noise
        kinds and two contamination protocols.

Data: outputs/ssh/main/sens_* (3 seeds each; epsilon_m = 1e-2 has no result
because the hierarchy saturates and exceeds the memory budget) and
outputs/ssh/main/noise_r0.2 (2 seeds each, paired reference produced by
scripts/evaluate_noise_paired.py).
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


GROUPS = [
    ("prototypes $K$", [
        ("k1", "1"), ("k2", "2"), ("k4", "4"), ("k16", "16"),
    ]),
    ("prototype temperature $\\tau_p$", [
        ("tau005", "0.05"), ("tau020", "0.20"), ("tau050", "0.50"),
    ]),
    ("prototype scale $\\lambda_p$", [
        ("lambda000", "0"), ("lambda025", "0.25"), ("lambda050", "0.5"),
        ("lambda100", "1.0"), ("lambda200", "2.0"),
    ]),
    ("boundary temperature $\\tau_b$", [
        ("tau005", "0.05"), ("tau025", "0.25"), ("tau050", "0.50"), ("tau100", "1.00"),
    ]),
    ("boundary rates $(r_A,r_E)$", [
        ("ra001_re0005", "0.01/0.005"), ("ra001_re0001", "0.01/0.001"),
        ("ra010_re0020", "0.10/0.02"), ("ra010_re0050", "0.10/0.05"),
    ]),
    ("temporal radius $R_l$", [("half", "half"), ("double", "double")]),
    ("semantic neighbours $K_l$", [("half", "half"), ("double", "double")]),
    ("membership threshold $\\epsilon_m$", [
        ("eps1e-8", "$10^{-8}$"), ("eps1e-4", "$10^{-4}$"), ("eps1e-2", "$10^{-2}$"),
    ]),
]

NOISE_KINDS = [("replace", "replace"), ("merge", "merge"), ("split", "split"), ("unk", "UNK")]
NOISE_PROTOCOLS = [("test_only", "test only"), ("all", "train + test")]


def seed_dirs(root: Path, tag: str, seeds: set[int] | None = None):
    for seed_dir in sorted((root / tag).glob("seed_*")):
        if seeds is not None and int(seed_dir.name.split("_")[1]) not in seeds:
            continue
        yield seed_dir


def stat(root: Path, tag: str, metric: str, seeds: set[int] | None = None):
    values = []
    for seed_dir in seed_dirs(root, tag, seeds):
        result = seed_dir / "result.json"
        if result.is_file():
            payload = json.loads(result.read_text(encoding="utf-8"))["test"]
            if payload.get(metric) is not None:
                values.append(float(payload[metric]))
    return (len(values), st.mean(values)) if values else None


def best_validation(root: Path, tag: str, seeds: set[int] | None = None):
    values = []
    for seed_dir in seed_dirs(root, tag, seeds):
        history = seed_dir / "history.json"
        if not history.is_file():
            continue
        try:
            records = json.loads(history.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if records:
            values.append(max(row["validation"]["auprc"] for row in records))
    return (len(values), st.mean(values)) if values else None


def paired_deltas(root: Path, tag: str, metric: str):
    """Corrupted-test minus clean-test, same weights and same threshold."""
    values = []
    for seed_dir in seed_dirs(root, tag):
        noisy_path = seed_dir / "result.json"
        clean_path = seed_dir / "clean_paired.json"
        if not (noisy_path.is_file() and clean_path.is_file()):
            continue
        noisy = json.loads(noisy_path.read_text(encoding="utf-8"))["test"].get(metric)
        clean = json.loads(clean_path.read_text(encoding="utf-8"))["clean_test"].get(metric)
        if noisy is None or clean is None:
            continue
        values.append(100.0 * (float(noisy) - float(clean)))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dataset", default="ssh")
    parser.add_argument("--reference", default="ladder_full/L7")
    parser.add_argument("--reference-seeds", nargs="+", type=int, default=[42, 123, 256],
                        help="seed subset of the reference run, matching the sweep")
    parser.add_argument("--noise-rate", default="0.2")
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_sensitivity_robustness")
    args = parser.parse_args()

    root = Path(args.output_root) / args.dataset / "main"
    reference_seeds = set(args.reference_seeds)
    reference_val = best_validation(root, args.reference, reference_seeds)
    reference_f1 = stat(root, args.reference, "f1", reference_seeds)
    if reference_val is None or reference_f1 is None:
        raise SystemExit("reference run is missing")

    rows = []
    for group, variants in GROUPS:
        rows.append(("group", group, None, None))
        for name, label in variants:
            tag = tag_for(group, name)
            validation = best_validation(root, tag)
            f1 = stat(root, tag, "f1")
            if validation is None and f1 is None:
                rows.append(("variant", label, None, None))
                continue
            delta_val = None if validation is None else validation[1] - reference_val[1]
            delta_f1 = None if f1 is None else f1[1] - reference_f1[1]
            rows.append(("variant", label, delta_val, delta_f1))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6),
                             gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.52})
    ax_sens, ax_noise = axes

    y = 0
    positions: list[tuple[str, int]] = []
    tick_labels: list[str] = []
    for kind, label, delta_val, delta_f1 in rows:
        if kind == "group":
            positions.append(("group", y))
            tick_labels.append(label)
            y += 1
            continue
        if delta_val is None and delta_f1 is None:
            ax_sens.text(0, y, "  (memory-infeasible: cell not run)", fontsize=5.4,
                         va="center", ha="left", color=PALETTE["neutral_mid"])
        else:
            if delta_val is not None:
                ax_sens.plot(delta_val * 100, y, marker="o", ms=3.6, mfc="white",
                             mec=PALETTE["blue_main"], mew=1.0, ls="none")
            if delta_f1 is not None:
                ax_sens.plot(delta_f1 * 100, y, marker="s", ms=3.6,
                             color=PALETTE["red_strong"], ls="none")
        positions.append(("variant", y))
        tick_labels.append("   " + label)
        y += 1
    ax_sens.axvline(0, color=PALETTE["neutral_dark"], lw=0.8)
    ax_sens.set_yticks([p for _, p in positions])
    ax_sens.set_yticklabels(tick_labels, fontsize=5.8)
    for (kind, position), tick in zip(positions, ax_sens.get_yticklabels()):
        if kind == "group":
            tick.set_fontsize(6.4)
            tick.set_color(PALETTE["neutral_black"])
    ax_sens.invert_yaxis()
    ax_sens.set_xlabel("change vs default (percentage points)")
    ax_sens.set_title("(a) hyper-parameter sensitivity", loc="left", fontsize=7.2)
    handles = [
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=PALETTE["blue_main"],
                   mew=1.0, ms=4.0, label="validation AUPRC (selection)"),
        plt.Line2D([], [], marker="s", ls="none", color=PALETTE["red_strong"], ms=4.0,
                   label="test F1 (reported)"),
    ]
    sensitivity_handles = handles

    # (b) parsing-noise robustness: paired with the same weights and threshold
    ax_auprc = ax_noise.twinx()
    ax_auprc.spines["right"].set_visible(True)
    ax_auprc.spines["top"].set_visible(False)
    kind_x = np.arange(len(NOISE_KINDS))
    width = 0.34
    f1_limits = (-6.0, 18.0)
    protocol_colours = [PALETTE["neutral_mid"], PALETTE["green_3"]]
    for position, (protocol, protocol_label) in enumerate(NOISE_PROTOCOLS):
        offset = (position - 0.5) * width
        means, low, high = [], [], []
        for kind, _ in NOISE_KINDS:
            values = paired_deltas(root, f"noise_r{args.noise_rate}/{kind}_{protocol}", "f1")
            if values:
                means.append(st.mean(values))
                low.append(st.mean(values) - min(values))
                high.append(max(values) - st.mean(values))
            else:
                means.append(np.nan)
                low.append(0.0)
                high.append(0.0)
        ax_noise.bar(kind_x + offset, means, width=width, color=protocol_colours[position],
                     alpha=0.85, edgecolor=PALETTE["neutral_dark"], linewidth=0.4,
                     label=protocol_label, zorder=2)
        ax_noise.errorbar(kind_x + offset, means, yerr=[low, high], fmt="none",
                          ecolor=PALETTE["neutral_black"], elinewidth=0.7, capsize=1.6,
                          zorder=3)
        for position_kind, (kind, _) in enumerate(NOISE_KINDS):
            values = paired_deltas(root, f"noise_r{args.noise_rate}/{kind}_{protocol}", "auprc")
            if not values:
                continue
            for step, value in enumerate(values):
                jitter = (step - (len(values) - 1) / 2) * 0.045
                ax_auprc.plot(kind_x[position_kind] + offset + jitter, value * 100,
                              marker="o", ms=3.0, ls="none", mfc="white",
                              mec=PALETTE["blue_main"], mew=0.8, zorder=4)
    ax_noise.axhline(0, color=PALETTE["neutral_dark"], lw=0.8, zorder=1)
    ax_noise.set_xticks(kind_x)
    ax_noise.set_xticklabels([label for _, label in NOISE_KINDS])
    ax_noise.set_ylim(*f1_limits)
    span = f1_limits[1] - f1_limits[0]
    ax_auprc.set_ylim(-f1_limits[0] / span * 2.4, f1_limits[1] / span * 2.4)
    ax_noise.set_xlabel("template-level parser corruption")
    ax_noise.set_ylabel("$\\Delta$F1, corrupted $-$ clean (pt)")
    ax_auprc.set_ylabel("$\\Delta$AUPRC (pt)", color=PALETTE["blue_main"])
    ax_auprc.tick_params(axis="y", colors=PALETTE["blue_main"])
    ax_auprc.tick_params(axis="x", labelbottom=False)
    ax_noise.set_title("(b) parsing-noise robustness", loc="left", fontsize=7.2)
    noise_handles = [
        plt.Rectangle((0, 0), 1, 1, color=protocol_colours[0], label="test-only"),
        plt.Rectangle((0, 0), 1, 1, color=protocol_colours[1], label="train + test"),
        plt.Line2D([], [], marker="o", ls="none", mfc="white",
                   mec=PALETTE["blue_main"], mew=0.9, ms=3.6, label="$\\Delta$AUPRC"),
    ]

    fig.legend(sensitivity_handles + noise_handles,
               [handle.get_label() for handle in sensitivity_handles + noise_handles],
               loc="lower center", ncol=5, fontsize=5.8,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.185, right=0.90, top=0.93, bottom=0.17)
    offenders = canvas_qa(fig) + legend_overlap_qa(fig) + text_overlap_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


def tag_for(group: str, name: str) -> str:
    mapping = {
        "prototypes $K$": "prototypes",
        "prototype temperature $\\tau_p$": "prototype_temperature",
        "prototype scale $\\lambda_p$": "prototype_scale",
        "boundary temperature $\\tau_b$": "boundary_temperature",
        "boundary rates $(r_A,r_E)$": "boundary_rate",
        "temporal radius $R_l$": "temporal_radius",
        "semantic neighbours $K_l$": "semantic_neighbors",
        "membership threshold $\\epsilon_m$": "membership_epsilon",
    }
    return f"sens_{mapping[group]}_{name}"


if __name__ == "__main__":
    main()
