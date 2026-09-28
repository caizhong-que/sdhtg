# -*- coding: utf-8 -*-
"""
_figure_common.py -- shared palette, rcParams, export and canvas QA helpers.

All paper figures are drawn with the Python/matplotlib backend so that the
vector exports keep editable text. The QA helper checks that every text artist
stays inside the canvas, which is the failure mode that cannot be caught by
looking at file sizes alone.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_soft": "#B4C0E4",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "red_soft": "#E9A6A1",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#D9A93B",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

METHOD_COLOR = {
    "TCN": PALETTE["blue_main"],
    "Transformer": PALETTE["teal"],
    "GNN-flat": PALETTE["violet"],
    "SDHTG": PALETTE["red_strong"],
}


def apply_style() -> None:
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


def save_pub(fig, out_dir: str | Path, name: str, dpi: int = 600) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / name
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    border_qa(f"{stem}.png")
    return stem


def canvas_qa(fig, tolerance: float = 1.0, verbose: bool = True) -> list[tuple]:
    """Report text artists whose bounding box leaves the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    offenders: list[tuple] = []
    texts = []
    for axes in fig.axes:
        texts.extend(axes.texts)
        texts.append(axes.title)
        if getattr(axes, "axison", True):
            texts.extend(axes.get_xticklabels())
            texts.extend(axes.get_yticklabels())
            texts.append(axes.xaxis.label)
            texts.append(axes.yaxis.label)
        legend = axes.get_legend()
        if legend is not None:
            texts.extend(legend.get_texts())
    for text in texts:
        if not text.get_text().strip() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer=renderer)
        if (
            box.x0 < canvas.x0 - tolerance
            or box.y0 < canvas.y0 - tolerance
            or box.x1 > canvas.x1 + tolerance
            or box.y1 > canvas.y1 + tolerance
        ):
            offenders.append(
                (text.get_text()[:32], round(box.x0), round(box.y0), round(box.x1), round(box.y1))
            )
    if verbose:
        state = "clean" if not offenders else f"{len(offenders)} overflow(s)"
        print(f"[qa] canvas {tuple(round(v) for v in canvas.bounds)}: {state}")
        for row in offenders:
            label = str(row[0]).encode("ascii", "replace").decode("ascii")
            print("     ", (label, *row[1:]))
    return offenders


def border_qa(png_path: str | Path, ink_threshold: float = 0.985, verbose: bool = True) -> int:
    """Count border pixels that carry ink in a saved PNG.

    The canvas check above flags labels that sit a few pixels outside the
    default figure rect, which ``bbox_inches="tight"`` simply expands to
    include. What would actually ruin a figure is content clipped by the saved
    boundary, so the saved raster is checked directly: a clean border means no
    artist was cut off.
    """
    import matplotlib.image as mpimg

    image = mpimg.imread(str(png_path))
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
    inked = int((border < ink_threshold).sum())
    if verbose:
        print(f"[qa] {Path(png_path).name}: {inked} inked border pixel(s) "
              f"{'(ok)' if inked == 0 else '(content may be clipped)'}")
    return inked


def legend_overlap_qa(fig, min_pixels: float = 4.0, verbose: bool = True) -> list[tuple]:
    """Report legends that overlap data artists inside their own axes.

    Full-width guide lines (``axhline`` / ``axvline``) are ignored: a legend
    placed over a light dotted guide is normal, whereas a legend sitting on top
    of bars, markers or curves is what readers complain about.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    offenders: list[tuple] = []
    for axes in fig.axes:
        legend = axes.get_legend()
        if legend is None or not legend.get_visible():
            continue
        legend_box = legend.get_window_extent(renderer=renderer)
        axes_extent = axes.get_window_extent(renderer=renderer)
        axes_area = max(axes_extent.width * axes_extent.height, 1.0)
        for artist in list(axes.lines) + list(axes.collections) + list(axes.patches):
            if getattr(artist, "_guide_line", None):
                continue
            try:
                box = artist.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if box.width <= 0 or box.height <= 0:
                continue
            # Errorbar containers and long curves have a bounding box spanning
            # most of the panel; only localised artists can genuinely be hidden.
            if box.width * box.height > 0.45 * axes_area:
                continue
            overlap = Bbox.intersection(legend_box, box)
            if overlap is None:
                continue
            if overlap.width < min_pixels or overlap.height < min_pixels:
                continue
            label = getattr(artist, "get_label", lambda: "")() or type(artist).__name__
            offenders.append(
                (axes.get_title()[:24] or "axes", label[:22],
                 round(overlap.width), round(overlap.height))
            )
    if verbose:
        print(f"[qa] legend/data overlaps: {len(offenders)}")
        for row in offenders:
            print(f"      {row}")
    return offenders


def text_overlap_qa(fig, min_width: float = 6.0, min_height: float = 5.0,
                    verbose: bool = True) -> list[tuple]:
    """Report text artists that overlap each other (titles, tick labels, notes).

    Only pairs whose intersection is larger than a small threshold are listed,
    so adjacent thin glyph boxes do not flood the report.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    entries: list[tuple[object, str, Bbox]] = []
    for axes in list(fig.axes) + [fig]:
        texts = []
        if axes is fig:
            texts = list(getattr(fig, "texts", []))
        else:
            texts = list(axes.texts) + [axes.title]
            if getattr(axes, "axison", True):
                texts += list(axes.get_xticklabels()) + list(axes.get_yticklabels())
                texts += [axes.xaxis.label, axes.yaxis.label]
        for text in texts:
            content = text.get_text().strip()
            if not content or not text.get_visible():
                continue
            try:
                box = text.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if box.width <= 0 or box.height <= 0:
                continue
            entries.append((text, content, box))

    offenders: list[tuple] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            object_a, label_a, box_a = entries[i]
            object_b, label_b, box_b = entries[j]
            if object_a is object_b:
                # Twin axes share the same tick label objects; comparing an
                # object with itself is not an overlap.
                continue
            overlap = Bbox.intersection(box_a, box_b)
            if overlap is None:
                continue
            if overlap.width < min_width or overlap.height < min_height:
                continue
            offenders.append(
                (label_a[:22].encode("ascii", "replace").decode("ascii"),
                 label_b[:22].encode("ascii", "replace").decode("ascii"),
                 round(overlap.width), round(overlap.height))
            )
    if verbose:
        print(f"[qa] text/text overlaps: {len(offenders)}")
        for row in offenders[:12]:
            print(f"      {row}")
    return offenders


# --------------------------------------------------------------------------
# result aggregation helpers (identical derivation to the manuscript tables)
# --------------------------------------------------------------------------

METHOD_TAGS = (
    ("baseline_tcn", "TCN"),
    ("baseline_transformer", "Transformer"),
    ("baseline_gnn_flat", "GNN-flat"),
    ("ladder_full/L7", "SDHTG"),
)


def load_results(output_root: str | Path, dataset: str, metric: str) -> dict[str, list[float]]:
    root = Path(output_root) / dataset / "main"
    values: dict[str, list[float]] = {}
    for tag, name in METHOD_TAGS:
        collected = []
        for seed_dir in sorted((root / tag).glob("seed_*")):
            result = seed_dir / "result.json"
            if result.is_file():
                payload = json.loads(result.read_text(encoding="utf-8"))["test"]
                if payload.get(metric) is not None:
                    collected.append(float(payload[metric]))
        if collected:
            values[name] = collected
    return values


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def test_positive_counts(processed_root: str | Path = "data/processed") -> dict[str, int]:
    counts = {}
    for dataset in ("bgl", "hdfs", "openstack", "ssh", "thunderbird"):
        report = Path(processed_root) / dataset / "quality_report.json"
        counts[dataset] = json.loads(report.read_text(encoding="utf-8"))["splits"]["test"][
            "anomalous_sessions"
        ]
    return counts
