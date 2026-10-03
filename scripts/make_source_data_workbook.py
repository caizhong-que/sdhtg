# -*- coding: utf-8 -*-
"""make_source_data_workbook.py -- build Source_Data_1.xlsx for the submission."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter


PARENT = Path(r"E:\SDHTG\文章\Download+the+journal+article+template+package+(December+2024+version)")
SUPP = PARENT / "supplementary"
TABLES = SUPP / "tables"
SOURCE = SUPP / "source_data"
TARGET = SUPP / "Source_Data_1.xlsx"


SHEETS = [
    ("S1 paired statistics", TABLES / "S1_paired_statistics.csv"),
    ("S2 main secondary metrics", TABLES / "S2_secondary_metrics_main.csv"),
    ("S3 imbalance prototype", TABLES / "S3_imbalance_prototype_seed_level.csv"),
    ("S5 sensitivity scan", TABLES / "S5_sensitivity_scan_seed_level.csv"),
    ("S6 semantic shortcuts", TABLES / "S6_semantic_shortcuts_seed_level.csv"),
    ("S7 parsing noise", TABLES / "S7_parsing_noise_seed_level.csv"),
    ("S8 entity identifiability", TABLES / "S8_entity_identifiability_seed_level.csv"),
    ("S9 masked-template baseline", TABLES / "S9_masked_template_baseline_seed_level.csv"),
    ("S12 negative sampling", TABLES / "S12_negative_sampling_seed_level.csv"),
    ("all runs per seed", SOURCE / "per_seed_metrics_all_runs.csv"),
]

README_ROWS = [
    ["Source Data 1", "Per-seed results underlying every table and figure of the manuscript."],
    ["", ""],
    ["Sheet", "Content"],
    ["S1 paired statistics", "Wilcoxon signed-rank tests with Holm-corrected q values and Cliff's delta, "
                             "grouped by comparison family (primary comparison, comparison against the flat GRU "
                             "baseline, module steps). Pairing unit = matched random seed."],
    ["S2 main secondary metrics", "AUROC, precision, recall, MCC and balanced accuracy for every dataset and "
                                  "method of the main comparison (mean and standard deviation over five seeds). "
                                  "MCC and balanced accuracy are derived from the reported precision, recall and "
                                  "the test-set class composition."],
    ["S3 imbalance prototype", "Seed-level AUPRC and F1 for the imbalance-learning and prototype configurations "
                               "on SSH (three seeds per configuration)."],
    ["S5 sensitivity scan", "Seed-level validation AUPRC, test AUPRC and test F1 for every hyper-parameter "
                            "variant of the sensitivity scan (78 runs)."],
    ["S6 semantic shortcuts", "Seed-level AUPRC and F1 for the semantic and entity shortcut controls on SSH "
                              "(five seeds per variant)."],
    ["S7 parsing noise", "Seed-level AUPRC and F1 on the corrupted test split and, for the same weights and "
                         "threshold, on the clean test split (rewrite rates 20% and 40%, four corruption kinds, "
                         "two contamination protocols, two seeds)."],
    ["S8 entity identifiability", "Seed-level group metrics for the entity-holdout protocol (seen, holdout and "
                                  "pooled groups) on BGL and OpenStack."],
    ["S9 masked-template baseline", "Seed-level AUPRC and F1 for the flat Transformer with and without "
                                    "masked-template pretraining and for SDHTG (SSH and OpenStack: five seeds; "
                                    "BGL: one seed)."],
    ["S12 negative sampling", "Seed-level AUPRC and F1 for the negative-sampling strategies used in contrastive "
                              "pretraining on SSH (three seeds per strategy and protocol)."],
    ["all runs per seed", "Every finished run in the repository at submission time: dataset, run tag, seed, "
                          "AUPRC, AUROC, precision, recall, F1, derived MCC and balanced accuracy, number of test "
                          "samples, calibrated threshold and training epochs."],
    ["", ""],
    ["Provenance", "Generated from outputs/<dataset>/main/**/seed_*/result.json plus the paired clean references "
                   "written by scripts/evaluate_noise_paired.py and the group files written by "
                   "scripts/evaluate_seen_unseen.py. The generating script is scripts/make_supplementary.py."],
    ["Software", "Python 3.11.15, PyTorch 2.5.1 (CUDA 11.8); see requirements-lock.txt."],
]


def main() -> None:
    with pd.ExcelWriter(TARGET, engine="openpyxl") as writer:
        readme = pd.DataFrame(README_ROWS, columns=["Item", "Description"])
        readme.to_excel(writer, sheet_name="README", index=False)
        for name, path in SHEETS:
            frame = pd.read_csv(path)
            frame.to_excel(writer, sheet_name=name[:31], index=False)

        book = writer.book
        for sheet in book.worksheets:
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for column in sheet.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0
                            for cell in column)
                sheet.column_dimensions[get_column_letter(column[0].column)].width = min(60, width + 2)
        readme_sheet = book["README"]
        readme_sheet.column_dimensions["A"].width = 28
        readme_sheet.column_dimensions["B"].width = 110
        for row in readme_sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    size_mb = TARGET.stat().st_size / 1e6
    print(f"wrote {TARGET} ({size_mb:.2f} MB)")
    print("sheets:", [name for name, _ in SHEETS])


if __name__ == "__main__":
    main()
