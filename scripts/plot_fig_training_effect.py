# -*- coding: utf-8 -*-
"""plot_fig_training_effect.py -- figure 8: training dynamics and effect sizes.

Panels:
    (a) validation AUPRC against the epoch for the five datasets (full model,
        five seeds; line = mean, band = min..max) -- shows that the reported
        operating points sit on flat, converged curves rather than a lucky epoch
    (b) the two curriculum schedules actually used by the trainer (boundary
        temperature cosine anneal and FiLM-strength warm-up, plus the boundary
        loss scale), reproduced from sdhtg.training.curriculum with the SSH
        experiment configuration
    (c) Cliff's delta of the paired comparisons (statistics_report.json):
        primary comparison and the flat baseline per dataset, F1 and AUPRC

Data: outputs/<dataset>/main/ladder_full/L7/seed_*/history.json and
outputs/statistics_report.json.

Usage:
    python scripts/plot_fig_training_effect.py --out-dir figure
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
import yaml


DATASETS = [("ssh", "SSH"), ("hdfs", "HDFS"), ("bgl", "BGL"),
            ("openstack", "OpenStack"), ("thunderbird", "Thunderbird")]
DATASET_COLOUR = {
    "SSH": PALETTE["red_strong"],
    "HDFS": PALETTE["blue_main"],
    "BGL": PALETTE["green_3"],
    "OpenStack": PALETTE["gold"],
    "Thunderbird": PALETTE["violet"],
}


def validation_curves(root: Path, dataset: str):
    runs = []
    for seed_dir in sorted((root / dataset / "main" / "ladder_full" / "L7").glob("seed_*")):
        history = seed_dir / "history.json"
        if history.is_file():
            runs.append(json.loads(history.read_text(encoding="utf-8")))
    if not runs:
        return None
    length = max(len(run) for run in runs)
    mean = np.full(length, np.nan)
    low = np.full(length, np.nan)
    high = np.full(length, np.nan)
    for epoch in range(length):
        available = [float(run[epoch]["validation"]["auprc"]) for run in runs
                     if epoch < len(run)]
        mean[epoch] = st.mean(available)
        low[epoch] = min(available)
        high[epoch] = max(available)
    return mean, low, high


def curriculum_curves(root: Path):
    """Actual schedules, read from the recorded training histories."""
    history = yaml.safe_load(
        Path("configs/experiment/ssh.yaml").read_text(encoding="utf-8"))["curriculum"]
    temperatures, films, per_dataset = [], [], {}
    for dataset, label in DATASETS:
        runs = []
        for seed_dir in sorted((root / dataset / "main" / "ladder_full" / "L7").glob("seed_*")):
            path = seed_dir / "history.json"
            if path.is_file():
                runs.append(json.loads(path.read_text(encoding="utf-8")))
        if not runs:
            continue
        temps = [np.asarray([float(row["train"]["temperature"]) for row in run])
                 for run in runs]
        film_block = [np.asarray([float(row["train"]["film_strength"]) for row in run])
                      for run in runs]
        temperatures.append(temps)
        films.append(film_block)
        per_dataset[label] = temps
    if not temperatures:
        return None

    def pooled(series_groups):
        flat = [series for group in series_groups for series in group]
        length = max(len(series) for series in flat)
        values = np.full(length, np.nan)
        for epoch in range(length):
            available = [series[epoch] for series in flat if epoch < len(series)]
            values[epoch] = st.mean(available)
        return values

    temperature = pooled(temperatures)
    film = pooled(films)
    epochs = np.arange(len(temperature))
    scale = np.minimum((epochs + 1) / max(int(history["boundary_loss_warmup_epochs"]), 1), 1.0)
    per_dataset_mean = {label: pooled([group]) for label, group in per_dataset.items()}
    return epochs, temperature, film, scale, per_dataset_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--out-dir", default="figure")
    parser.add_argument("--name", default="fig_training_effect")
    args = parser.parse_args()

    root = Path(args.output_root)
    report_path = root / "statistics_report.json"
    if not report_path.is_file():
        raise SystemExit("run scripts/statistics_report.py first")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    apply_style()
    fig = plt.figure(figsize=(7.4, 2.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.15], wspace=0.42)
    ax_curve, ax_schedule, ax_effect = (fig.add_subplot(grid[0, i]) for i in range(3))

    # ---------------------------------------------------------------- panel a
    curve_handles = []
    for dataset, label in DATASETS:
        curves = validation_curves(root, dataset)
        if curves is None:
            continue
        mean, low, high = curves
        epochs = np.arange(len(mean))
        line, = ax_curve.plot(epochs, mean, color=DATASET_COLOUR[label], lw=1.1,
                              label=label)
        curve_handles.append(line)
        ax_curve.fill_between(epochs, low, high,
                              color=DATASET_COLOUR[label], alpha=0.16, lw=0)
    ax_curve.set_xlabel("training epoch")
    ax_curve.set_ylabel("validation AUPRC")
    ax_curve.set_title("(a) validation AUPRC of the full model", loc="left", fontsize=7.2)
    ax_curve.set_xlim(-0.6, 30.6)
    ax_curve.set_ylim(0.50, 1.012)
    ax_curve.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax_curve.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_curve.annotate("HDFS seed 42\nlate-epoch collapse",
                      xy=(23.0, 0.60), xytext=(11.5, 0.545), fontsize=4.8,
                      color=PALETTE["neutral_black"], ha="left", va="center",
                      arrowprops=dict(arrowstyle="-", lw=0.5,
                                      color=PALETTE["neutral_mid"],
                                      shrinkA=0.0, shrinkB=1.5))

    # ---------------------------------------------------------------- panel b
    epochs, temperature, film, scale, per_dataset = curriculum_curves(root)
    for label, series in per_dataset.items():
        ax_schedule.plot(np.arange(len(series)), series, color=PALETTE["blue_soft"],
                         lw=0.6, alpha=0.75, zorder=1)
    ax_schedule.plot(epochs, temperature, color=PALETTE["blue_main"], lw=1.3,
                     zorder=3)
    for label, x_position, y_position in (("SSH 0.10", 30.5, 0.10),
                                          ("HDFS 0.10", 25.0, 0.035),
                                          ("Thunderbird 0.18", 25.5, 0.24),
                                          ("BGL 0.39", 20.5, 0.47),
                                          ("OpenStack 0.39", 20.5, 0.31)):
        ax_schedule.text(x_position, y_position, label, fontsize=4.8,
                         color=PALETTE["blue_main"], va="center", ha="left")
    ax_schedule.text(1.0, 0.06, "boundary-loss warm-up (5 epochs)", fontsize=4.8,
                     color=PALETTE["neutral_black"], va="center", ha="left")
    ax_schedule.set_xlabel("training epoch")
    ax_schedule.set_ylabel("boundary temperature $\\tau_b$")
    ax_schedule.set_ylim(-0.03, 1.06)
    ax_schedule.set_xlim(-0.6, 38)
    ax_schedule.set_xticks([0, 10, 20, 30])
    ax_schedule.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_schedule.set_title("(b) boundary-temperature anneal", loc="left", fontsize=7.2)
    schedule_handle = plt.Line2D([], [], color=PALETTE["blue_main"], lw=1.3,
                                 label="mean $\\tau_b$ (five datasets)")

    # ---------------------------------------------------------------- panel c
    matrix, columns = [], []
    for dataset, label in DATASETS:
        columns.append(label)
    for family, metric, title in (("primary", "f1", "primary F1"),
                                  ("primary", "auprc", "primary AUPRC"),
                                  ("baseline", "f1", "vs L0 F1")):
        row = []
        for dataset, _ in DATASETS:
            value = next((r for r in report[family]
                          if r["dataset"] == dataset and r["metric"] == metric), None)
            row.append(np.nan if value is None else value["cliff"])
        matrix.append(row)
    matrix = np.asarray(matrix, dtype=float)
    image = ax_effect.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax_effect.set_xticks(np.arange(len(columns)))
    ax_effect.set_xticklabels(columns, fontsize=5.2, rotation=42, ha="right",
                              rotation_mode="anchor")
    ax_effect.set_yticks(np.arange(len(matrix)))
    ax_effect.set_yticklabels(["primary F1", "primary AUPRC", "vs L0 F1"], fontsize=5.6)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                continue
            ax_effect.text(column, row, f"{value:+.2f}", ha="center", va="center",
                           fontsize=5.0,
                           color="white" if abs(value) > 0.55 else PALETTE["neutral_black"])
    ax_effect.set_title("(c) Cliff's $\\delta$ vs baselines", loc="left", fontsize=7.2)
    bar = fig.colorbar(image, ax=ax_effect, fraction=0.045, pad=0.02)
    bar.ax.tick_params(labelsize=5.2)
    bar.outline.set_linewidth(0.4)

    fig.legend(curve_handles + [schedule_handle],
               [handle.get_label() for handle in curve_handles] + [schedule_handle.get_label()],
               loc="lower center", ncol=6, fontsize=5.4, frameon=False,
               bbox_to_anchor=(0.5, 0.002), handlelength=1.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.075, right=0.915, top=0.885, bottom=0.235)
    offenders = canvas_qa(fig) + legend_overlap_qa(fig) + text_overlap_qa(fig)
    stem = save_pub(fig, args.out_dir, args.name)
    print(f"wrote {stem}.svg / .pdf / .png ({len(offenders)} canvas overflows)")


if __name__ == "__main__":
    main()
