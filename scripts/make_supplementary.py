# -*- coding: utf-8 -*-
"""make_supplementary.py -- build the supplementary package for the final paper.

Writes machine-readable tables plus a ready-to-paste LaTeX appendix into the
directory *above* the Springer Nature template folder.  Nothing inside the
template folder is touched.

Sources: outputs/**/result.json (per-seed metrics), the paired clean
references written by evaluate_noise_paired.py, the seen/unseen group JSONs,
outputs/statistics_report.json and the processed-cache manifests.

Usage:
    python scripts/make_supplementary.py
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import pandas as pd


PARENT = Path(r"E:\SDHTG\文章\Download+the+journal+article+template+package+(December+2024+version)")
TARGET = PARENT / "supplementary"
TABLES = TARGET / "tables"
FIGURES = TARGET / "figures"
SOURCE = TARGET / "source_data"
PROTOCOL = TARGET / "protocol"
FIGURE_SRC = Path(r"E:\SDHTG\文章\初稿\figure")

METHOD_NAMES = {
    "baseline_tcn": "TCN",
    "baseline_transformer": "Transformer",
    "baseline_gnn_flat": "GNN-flat",
    "ladder_full/L0": "GRU-flat (L0)",
    "ladder_full/L1": "L1 +FiLM",
    "ladder_full/L2": "L2 +action boundary/hierarchy",
    "ladder_full/L3": "L3 +entity boundary",
    "ladder_full/L4": "L4 +temporal edges",
    "ladder_full/L5": "L5 +semantic edges",
    "ladder_full/L6": "L6 +cross-level messages",
    "ladder_full/L7": "SDHTG (L7)",
    "ladder_full/L7b": "L7b +prototype diversity/balance",
    "mlm_baseline": "Transformer + masked-template pretraining",
}

DATASETS = ["bgl", "hdfs", "openstack", "ssh", "thunderbird"]


def run_records() -> list[dict]:
    rows = []
    for path in sorted(Path("outputs").glob("*/main/**/seed_*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        test = payload.get("test") or {}
        dataset = path.parts[1]
        seed = int(path.parent.name.split("_")[1])
        tag = path.parent.parent.relative_to(Path("outputs") / dataset / "main").as_posix()
        rows.append({
            "dataset": dataset,
            "run_tag": tag,
            "seed": seed,
            "auprc": test.get("auprc"),
            "auroc": test.get("auroc"),
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "f1": test.get("f1"),
            "samples": test.get("samples"),
            "threshold": test.get("threshold_applied"),
            "epochs": payload.get("epochs"),
        })
    return rows


def test_composition(dataset: str) -> tuple[int, int]:
    frame = pd.read_parquet(f"data/processed/{dataset}/sessions.parquet",
                            columns=["split", "label"])
    test = frame[frame.split == "test"]
    positives = int(test.label.sum())
    return positives, int(len(test) - positives)


def derive_mcc_ba(precision, recall, positives, negatives):
    if not precision or not recall or precision <= 0:
        return None, None
    tp = recall * positives
    fn = positives - tp
    fp = tp / precision - tp
    tn = negatives - fp
    if min(tp, fp, fn, tn) < 0:
        return None, None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else None
    ba = 0.5 * (tp / max(tp + fn, 1e-9) + tn / max(tn + fp, 1e-9))
    return mcc, ba


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    for folder in (TABLES, FIGURES, SOURCE, PROTOCOL):
        folder.mkdir(parents=True, exist_ok=True)

    records = run_records()
    print(f"collected {len(records)} runs")

    # ---------------------------------------------- all runs, per-seed metrics
    header = ["dataset", "run_tag", "seed", "auprc", "auroc", "precision", "recall",
              "f1", "mcc", "balanced_accuracy", "samples", "threshold", "epochs"]
    rows = []
    for record in records:
        positives, negatives = test_composition(record["dataset"])
        mcc, ba = derive_mcc_ba(record["precision"], record["recall"], positives, negatives)
        rows.append([record["dataset"], record["run_tag"], record["seed"],
                     record["auprc"], record["auroc"], record["precision"], record["recall"],
                     record["f1"], mcc, ba, record["samples"], record["threshold"],
                     record["epochs"]])
    write_csv(SOURCE / "per_seed_metrics_all_runs.csv", header, rows)

    # ------------------------------------------- S2 secondary metric summary
    summary = []
    for dataset in DATASETS:
        for tag in ("baseline_tcn", "baseline_transformer", "baseline_gnn_flat",
                    "ladder_full/L0", "ladder_full/L7"):
            subset = [r for r in rows if r[0] == dataset and r[1] == tag]
            if not subset:
                continue
            def stat(index):
                values = [row[index] for row in subset if row[index] is not None]
                if not values:
                    return "--"
                mean = sum(values) / len(values)
                sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 \
                    if len(values) > 1 else 0.0
                return f"{mean:.4f} ± {sd:.4f}"
            summary.append([dataset, METHOD_NAMES[tag], len(subset),
                            stat(3), stat(4), stat(5), stat(6), stat(7), stat(8), stat(9)])
    write_csv(TABLES / "S2_secondary_metrics_main.csv",
              ["dataset", "method", "seeds", "AUPRC", "AUROC", "Precision", "Recall",
               "F1", "MCC", "Balanced accuracy"], summary)

    # ------------------------------------------------- S3 imbalance / prototype
    imbalance = []
    for path in sorted(Path("outputs/ssh/main").glob("ablation_imbalance_*/seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))["test"]
        imbalance.append([path.parent.parent.name.replace("ablation_imbalance_", ""),
                          int(path.parent.name.split("_")[1]),
                          payload["auprc"], payload["f1"]])
    write_csv(TABLES / "S3_imbalance_prototype_seed_level.csv",
              ["configuration", "seed", "AUPRC", "F1"], imbalance)

    # ----------------------------------------------------- S5 sensitivity scan
    sensitivity = []
    for path in sorted(Path("outputs/ssh/main").glob("sens_*/seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sensitivity.append([path.parent.parent.name.replace("sens_", ""),
                            int(path.parent.name.split("_")[1]),
                            payload["best_metric"], payload["test"]["auprc"],
                            payload["test"]["f1"]])
    write_csv(TABLES / "S5_sensitivity_scan_seed_level.csv",
              ["setting", "seed", "validation_AUPRC", "test_AUPRC", "test_F1"], sensitivity)

    # ------------------------------------------------- S6 semantic shortcuts
    shortcuts = []
    for path in sorted(Path("outputs/ssh/main").glob("ablation_shortcuts_*/seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))["test"]
        shortcuts.append([path.parent.parent.name.replace("ablation_shortcuts_", ""),
                          int(path.parent.name.split("_")[1]),
                          payload["auprc"], payload["f1"]])
    write_csv(TABLES / "S6_semantic_shortcuts_seed_level.csv",
              ["variant", "seed", "AUPRC", "F1"], shortcuts)

    # ------------------------------------------------- S7 parsing-noise pairs
    noise = []
    for rate in ("0.2", "0.4"):
        for path in sorted(Path(f"outputs/ssh/main/noise_r{rate}").glob("*/seed_*/result.json")):
            tag = path.parent.parent.name
            seed = int(path.parent.name.split("_")[1])
            noisy = json.loads(path.read_text(encoding="utf-8"))["test"]
            clean_path = path.parent / "clean_paired.json"
            clean = json.loads(clean_path.read_text(encoding="utf-8"))["clean_test"] \
                if clean_path.is_file() else {}
            noise.append([rate, tag, seed,
                          noisy["auprc"], clean.get("auprc"), noisy["f1"], clean.get("f1")])
    write_csv(TABLES / "S7_parsing_noise_seed_level.csv",
              ["rewrite_rate", "condition", "seed", "AUPRC_noisy", "AUPRC_clean",
               "F1_noisy", "F1_clean"], noise)

    # ------------------------------------------- S8 entity identifiability
    entity = []
    for dataset in ("bgl", "openstack"):
        for path in sorted(Path(f"outputs/{dataset}/main/seen_unseen").glob("entholdout20_L7_seed_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for group in ("seen", "holdout", "all"):
                payload_group = payload["groups"].get(group)
                if not payload_group:
                    continue
                entity.append([dataset, payload["seed"], group,
                               payload_group["samples"], payload_group["positives"],
                               payload_group["auprc"], payload_group["f1"],
                               payload_group["precision"], payload_group["recall"]])
    write_csv(TABLES / "S8_entity_identifiability_seed_level.csv",
              ["dataset", "seed", "group", "samples", "positives", "AUPRC", "F1",
               "Precision", "Recall"], entity)

    # ----------------------------------------------- S9 masked-template baseline
    mlm = []
    for dataset in ("ssh", "openstack", "bgl"):
        for tag, label in (("baseline_transformer", "Transformer (no pretraining)"),
                           ("mlm_baseline", "Transformer + masked-template pretraining"),
                           ("ladder_full/L7", "SDHTG")):
            for path in sorted(Path(f"outputs/{dataset}/main/{tag}").glob("seed_*/result.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))["test"]
                mlm.append([dataset, label, int(path.parent.name.split("_")[1]),
                            payload["auprc"], payload["f1"]])
    write_csv(TABLES / "S9_masked_template_baseline_seed_level.csv",
              ["dataset", "method", "seed", "AUPRC", "F1"], mlm)

    # ---------------------------------------------- S12 negative sampling study
    negative = []
    for path in sorted(Path("outputs/ssh/main").glob("pretrain_*/seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))["test"]
        negative.append([path.parent.parent.name.replace("pretrain_", ""),
                         int(path.parent.name.split("_")[1]),
                         payload["auprc"], payload["f1"]])
    write_csv(TABLES / "S12_negative_sampling_seed_level.csv",
              ["protocol", "seed", "AUPRC", "F1"], negative)

    # -------------------------------------------------- source data & protocol
    for name in ("statistics_report.json", "efficiency_benchmark.json"):
        source = Path("outputs") / name
        if source.is_file():
            shutil.copy(source, SOURCE / name)
            print(f"  copied {name}")
    design = Path("EXPERIMENT_DESIGN.md")
    if design.is_file():
        shutil.copy(design, PROTOCOL / "EXPERIMENT_DESIGN.md")
        print("  copied EXPERIMENT_DESIGN.md")

    provenance = []
    for dataset in DATASETS:
        manifest = Path(f"data/processed/{dataset}/manifest.json")
        if not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        sessions = [o["sha256"] for o in payload["outputs"]
                    if o["path"].endswith("sessions.parquet")]
        provenance.append([dataset, payload["created_utc"], payload["git_commit"],
                           sessions[0] if sessions else "", payload["source_files"][0]["sha256"]])
    write_csv(PROTOCOL / "data_provenance_and_hashes.csv",
              ["dataset", "cache_built_utc", "git_commit", "sessions_parquet_sha256",
               "raw_log_sha256"], provenance)

    # ---------------------------------------------------------------- figures
    for source, target in (
        (FIGURE_SRC / "fig_training_effect", "SuppFig1_training_dynamics"),
        (FIGURE_SRC / "fig_sensitivity_robustness", "SuppFig2_sensitivity_noise_robustness"),
    ):
        for suffix in (".pdf", ".png", ".svg"):
            if source.with_suffix(suffix).is_file():
                shutil.copy(source.with_suffix(suffix), FIGURES / f"{target}{suffix}")
        print(f"  copied {target}")

    print("supplementary package written to " + str(TARGET))


if __name__ == "__main__":
    main()
