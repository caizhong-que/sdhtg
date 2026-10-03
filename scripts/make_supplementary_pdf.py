# -*- coding: utf-8 -*-
"""make_supplementary_pdf.py -- build Supplementary_Information.tex.

Assembles one standalone SI document (cover page, S1-S13, Figures S1-S2,
Supplementary Text S1-S2, Supplementary References) from the machine-readable
tables in ``supplementary/tables`` plus the two supplementary figures. Per-seed
detail stays in the Source Data workbook.

Usage:
    python scripts/make_supplementary_pdf.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd


PARENT = Path(r"E:\SDHTG\文章\Download+the+journal+article+template+package+(December+2024+version)")
SUPP = PARENT / "supplementary"
TABLES = SUPP / "tables"
FIGURES = SUPP / "figures"


def esc(text: object) -> str:
    out = str(text)
    out = out.replace("\\", "\\textbackslash{}")
    for source, target in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                           ("±", "$\\pm$"), ("≥", "$\\geq$"), ("≤", "$\\leq$"),
                           ("→", "$\\rightarrow$"), ("'", "'")):
        out = out.replace(source, target)
    return out


def table(header: list[str], rows: list[list], spec: str, caption: str,
          label: str, size: str = "\\small", resize: bool = False,
          note: str = "") -> str:
    """Emit one supplementary table (single page, forced position)."""
    lines = ["\\begin{table}[H]", "\\centering",
             f"\\caption*{{\\textbf{{Supplementary Table {label}.}} {caption}}}",
             f"{{{size}"]
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
              " & ".join(header) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if resize:
        lines.append("}")
    if note:
        lines.append("\\\\[2pt] \\footnotesize " + note)
    lines += ["}", "\\end{table}"]
    return "\n".join(lines)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def plus(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    stats = pd.read_csv(TABLES / "S1_paired_statistics.csv")
    secondary = pd.read_csv(TABLES / "S2_secondary_metrics_main.csv")
    imbalance = pd.read_csv(TABLES / "S3_imbalance_prototype_seed_level.csv")
    hyper = pd.read_csv(TABLES / "S4_hyperparameters.csv")
    sensitivity = pd.read_csv(TABLES / "S5_sensitivity_scan_seed_level.csv")
    shortcuts = pd.read_csv(TABLES / "S6_semantic_shortcuts_seed_level.csv")
    noise = pd.read_csv(TABLES / "S7_parsing_noise_seed_level.csv")
    entity = pd.read_csv(TABLES / "S8_entity_identifiability_seed_level.csv")
    mlm = pd.read_csv(TABLES / "S9_masked_template_baseline_seed_level.csv")
    literature = pd.read_csv(TABLES / "S10_literature_reference_values.csv")
    taxonomy = pd.read_csv(TABLES / "S11_baseline_taxonomy.csv")
    negative = pd.read_csv(TABLES / "S12_negative_sampling_seed_level.csv")
    semantic = pd.read_csv(TABLES / "S13_semantic_field_sources.csv")

    blocks: list[str] = []

    # ------------------------------------------------------------------ S1
    rows = []
    for family, group in stats.groupby("family", sort=False):
        rows.append(["\\multicolumn{4}{l}{\\textit{" + esc(family) + "}}",
                     "", "", "", "", ""])
        for _, row in group.iterrows():
            rows.append([
                esc(row["dataset"]), esc(row["comparison"]),
                "AUPRC" if row["metric"] == "auprc" else "F1",
                f"{int(row['paired_seeds'])}", plus(row["delta_pt"]),
                fmt(row["p_wilcoxon"]), fmt(row["q_holm"]), plus(row["cliff_delta"], 2),
                "$+$" if row["cliff_delta"] > 0 else "$-$",
            ])
    blocks.append(table(
        ["Dataset", "Comparison", "Metric", "Pairs", "$\\Delta$ (pt)", "$p$",
         "$q_{\\mathrm{Holm}}$", "Cliff's $\\delta$", "Direction"],
        rows, "llccrrrrl",
        ("Paired statistical comparisons. Wilcoxon signed-rank tests use the same random "
         "seed as the pairing unit; $q_{\\mathrm{Holm}}$ values are corrected within each "
         "family and metric, and Cliff's $\\delta$ is the effect size (positive = the first "
         "method is better). With five paired seeds the smallest attainable two-sided "
         "$p$ is 0.0625."),
        "S1", resize=True))

    # ------------------------------------------------------------------ S2
    rows = [[esc(r["dataset"]), esc(r["method"]), int(r["seeds"]), esc(r["AUPRC"]),
             esc(r["AUROC"]), esc(r["Precision"]), esc(r["Recall"]), esc(r["F1"]),
             esc(r["MCC"]), esc(r["Balanced accuracy"])] for _, r in secondary.iterrows()]
    blocks.append(table(
        ["Dataset", "Method", "Seeds", "AUPRC", "AUROC", "Precision", "Recall", "F1",
         "MCC", "Balanced accuracy"],
        rows, "llcrrrrrrr",
        ("Additional metrics for the main comparison (mean $\\pm$ standard deviation over "
         "the five matched random seeds). MCC and balanced accuracy are computed from the "
         "reported precision, recall and the test-set class composition."),
        "S2", size="\\scriptsize", resize=True))

    # ------------------------------------------------------------------ S3
    rows = []
    for configuration, group in imbalance.groupby("configuration", sort=False):
        rows.append([esc(configuration), len(group),
                     f"{group['AUPRC'].mean():.4f} $\\pm$ {group['AUPRC'].std(ddof=1):.4f}",
                     f"{group['F1'].mean():.4f} $\\pm$ {group['F1'].std(ddof=1):.4f}"])
    blocks.append(table(
        ["Configuration", "Seeds", "AUPRC", "F1"], rows, "lcrr",
        ("Imbalance-learning and prototype configurations on SSH. Each configuration is "
         "trained with the same protocol as the main experiments (30 epochs, three seeds); "
         "the corresponding cross-dataset incremental ladder is reported in the main text."),
        "S3"))

    # ------------------------------------------------------------------ S4
    rows = [[esc(r["symbol"]), esc(r["meaning"]), esc(r["value in final model"]),
             esc(r["sensitivity-scan values"]), esc(r["verified against"])]
            for _, r in hyper.iterrows()]
    blocks.append(table(
        ["Symbol", "Meaning", "Final value", "Sensitivity scan", "Verified against"],
        rows, "p{1.2cm}p{3.4cm}p{2.6cm}p{3.4cm}p{4.2cm}",
        ("Final hyper-parameters and implementation settings, with the configuration key "
         "that carries each value. Parameters without a sensitivity scan are fixed at the "
         "stated value."),
        "S4", size="\\scriptsize", resize=True))

    # ------------------------------------------------------------------ S5
    default = {"validation_AUPRC": 0.9853, "test_AUPRC": 0.9941, "test_F1": 0.9561}
    rows = []
    for setting, group in sensitivity.groupby("setting", sort=True):
        rows.append([
            esc(setting.replace("_", "\\_")), len(group),
            fmt(group["validation_AUPRC"].mean()),
            fmt(group["test_AUPRC"].mean()), fmt(group["test_F1"].mean()),
            plus((group["test_F1"].mean() - default["test_F1"]) * 100),
        ])
    blocks.append(table(
        ["Setting (group\\_variant)", "Seeds", "Validation AUPRC", "Test AUPRC", "Test F1",
         "$\\Delta$F1 vs default (pt)"],
        rows, "lcrrrr",
        ("Hyper-parameter sensitivity scan on SSH. Each variant is trained under the "
         "default configuration with one parameter changed; the reference is the default "
         "configuration evaluated on the same seeds (validation AUPRC 0.9853, test AUPRC "
         "0.9941, test F1 0.9561). Per-seed values are provided in Source Data 1."),
        "S5", size="\\scriptsize", resize=True))

    # ------------------------------------------------------------------ S6
    rows = []
    for variant, group in shortcuts.groupby("variant", sort=True):
        rows.append([esc(variant), len(group),
                     f"{group['AUPRC'].mean():.4f} $\\pm$ {group['AUPRC'].std(ddof=1):.4f}",
                     f"{group['F1'].mean():.4f} $\\pm$ {group['F1'].std(ddof=1):.4f}"])
    rows.append(["full input (reference)", 5, "$0.9922 \\pm 0.0031$", "$0.9565 \\pm 0.0183$"])
    blocks.append(table(
        ["Variant", "Seeds", "AUPRC", "F1"], rows, "lcrr",
        ("Semantic and entity shortcut controls on SSH: each row removes or alters one "
         "selected cue (entity embedding, boundary priors, action/state sources, entity "
         "identity) while keeping the remaining inputs unchanged. The last row repeats the "
         "full-input reference."),
        "S6"))

    # ------------------------------------------------------------------ S7
    rows = []
    for (rate, condition), group in noise.groupby(["rewrite_rate", "condition"], sort=True):
        d_auprc = (group["AUPRC_noisy"].mean() - group["AUPRC_clean"].mean()) * 100
        d_f1 = (group["F1_noisy"].mean() - group["F1_clean"].mean()) * 100
        kind, protocol = condition.rsplit("_", 1)
        rows.append([f"${int(rate * 100)}\\%$", esc(kind), esc(protocol), len(group),
                     fmt(group["AUPRC_noisy"].mean()), fmt(group["AUPRC_clean"].mean()),
                     plus(d_auprc), fmt(group["F1_noisy"].mean()),
                     fmt(group["F1_clean"].mean()), plus(d_f1)])
    blocks.append(table(
        ["Rate", "Corruption", "Protocol", "Seeds", "AUPRC$_{\\mathrm{noisy}}$",
         "AUPRC$_{\\mathrm{clean}}$", "$\\Delta$AUPRC", "F1$_{\\mathrm{noisy}}$",
         "F1$_{\\mathrm{clean}}$", "$\\Delta$F1"],
        rows, "lllcrrrrrr",
        ("Paired evaluation under template-level parsing noise (SSH). Replacement, merge, "
         "split and UNK-injection corruptions are applied at rewrite rates of 20\\% and "
         "40\\%; ``test\\_only'' corrupts the test split, ``all'' corrupts training, "
         "validation and test. Model parameters and the decision threshold calibrated on "
         "the clean validation split are kept fixed, and the clean column re-evaluates the "
         "same weights on the clean test split. $\\Delta$ is in percentage points."),
        "S7", size="\\scriptsize", resize=True))

    # ------------------------------------------------------------------ S8
    rows = []
    for (dataset, group_name), group in entity.groupby(["dataset", "group"], sort=True):
        rows.append([esc(dataset), esc(group_name), len(group), int(group["samples"].mean()),
                     int(group["positives"].mean()),
                     "--" if group["AUPRC"].isna().all() else fmt(group["AUPRC"].mean()),
                     fmt(group["F1"].mean()), fmt(group["Precision"].mean()),
                     fmt(group["Recall"].mean())])
    blocks.append(table(
        ["Dataset", "Group", "Seeds", "Samples", "Positives", "AUPRC", "F1", "Precision",
         "Recall"],
        rows, "llcccrrrr",
        ("Entity-identifiability evaluation. The entity identities covering about 20\\% of "
         "the test sessions are masked to a placeholder in training, validation and test, "
         "and the model is retrained; ``seen'' and ``holdout'' then refer to the same "
         "weights and the same calibrated threshold. OpenStack has no negatives in the "
         "holdout group, so AUPRC is undefined there."),
        "S8"))

    # ------------------------------------------------------------------ S9
    rows = []
    for (dataset, method), group in mlm.groupby(["dataset", "method"], sort=True):
        rows.append([esc(dataset), esc(method), len(group),
                     f"{group['AUPRC'].mean():.4f}"
                     + (f" $\\pm$ {group['AUPRC'].std(ddof=1):.4f}" if len(group) > 1 else ""),
                     f"{group['F1'].mean():.4f}"
                     + (f" $\\pm$ {group['F1'].std(ddof=1):.4f}" if len(group) > 1 else "")])
    blocks.append(table(
        ["Dataset", "Method", "Seeds", "AUPRC", "F1"], rows, "llcrr",
        ("Masked-template pretraining baseline under the controlled input protocol. The "
         "baseline shares the inputs, encoder depth and width, supervised objective, "
         "early-stopping rule and threshold calibration with the flat Transformer; only "
         "the 15\\% masked-template pretraining stage is added. SSH and OpenStack use five "
         "seeds, BGL one (it is saturated)."),
        "S9"))

    # ------------------------------------------------------------------ S10
    rows = []
    for _, r in literature.iterrows():
        def cell(value):
            return "--" if pd.isna(value) else f"{value:.3f}"
        rows.append([esc(r["source"]), esc(r["method"]), cell(r["HDFS"]), cell(r["BGL"]),
                     cell(r["Thunderbird"]), cell(r["OpenStack"]), cell(r["SSH"]),
                     esc(r["metric"]), esc(r["protocol note"])])
    blocks.append(table(
        ["Source", "Method", "HDFS", "BGL", "Thunderbird", "OpenStack", "SSH", "Metric",
         "Protocol note"],
        rows, "llccccclp{4.4cm}",
        ("Reference values reported by prior studies under their own protocols. "
         "Differences in preprocessing, split construction, label definition, threshold "
         "selection and evaluation protocol prevent direct numerical comparison with the "
         "controlled results of this work; the table is provided for context only."),
        "S10", size="\\scriptsize", resize=True))

    # ------------------------------------------------------------------ S11
    rows = [[esc(r["category"]), esc(r["representative methods"]),
             esc(r["original protocol (input / label unit)"]), esc(r["role in this work"])]
            for _, r in taxonomy.iterrows()]
    blocks.append(table(
        ["Category", "Representative methods", "Original protocol", "Role in this work"],
        rows, "p{2.4cm}p{3.4cm}p{5.2cm}p{3.4cm}",
        ("Baseline taxonomy and comparison protocol. ``Controlled'' baselines are "
         "implemented and trained in this work under identical inputs, splits, seeds and "
         "threshold selection; literature references are quoted and never merged into the "
         "controlled comparisons."),
        "S11", size="\\scriptsize", resize=True))

    # ------------------------------------------------------------------ S12
    rows = []
    for protocol_name, group in negative.groupby("protocol", sort=True):
        rows.append([esc(protocol_name), len(group),
                     f"{group['AUPRC'].mean():.4f} $\\pm$ {group['AUPRC'].std(ddof=1):.4f}",
                     f"{group['F1'].mean():.4f} $\\pm$ {group['F1'].std(ddof=1):.4f}"])
    blocks.append(table(
        ["Protocol (sampling \\_ data range)", "Seeds", "AUPRC", "F1"], rows, "lcrr",
        ("Negative-sampling strategies for contrastive pretraining on SSH. ``normal\\_only'' "
         "pretrains on normal sessions, ``all\\_train'' also uses unlabelled anomalous "
         "sessions; the suffix names the negative-selection rule (random, hardest, "
         "semi-hard, semantically filtered, no negatives, supervised contrastive). The "
         "reference value without pretraining is AUPRC $0.9922 \\pm 0.0031$."),
        "S12"))

    # ------------------------------------------------------------------ S13
    rows = [[esc(r["field"]), esc(r["construction"]), esc(r["source of the labels"]),
             esc(r["leakage control"])] for _, r in semantic.iterrows()]
    blocks.append(table(
        ["Field", "Construction", "Source of the labels", "Leakage control"],
        rows, "p{1.8cm}p{4.6cm}p{4.4cm}p{4.6cm}",
        ("Sources of the semantic fields and the controls used to prevent label or "
         "test-set leakage. No field is derived from anomaly labels, test-split templates, "
         "large language models or external anomaly knowledge bases; unknown values map to "
         "placeholders."),
        "S13", size="\\footnotesize", resize=True))

    body = "\n\n".join(blocks)
    figures = "\n".join([
        "\\begin{figure}[htbp]",
        "\\centering",
        "\\includegraphics[width=\\textwidth]{figures/SuppFig1_training_dynamics.pdf}",
        "\\caption*{\\textbf{Supplementary Figure S1.} Training dynamics and paired effect "
        "sizes. (a) Validation AUPRC against the training epoch for the five datasets "
        "(five seeds; line = mean, band = range). (b) The boundary-temperature anneal "
        "recorded during training. (c) Cliff's $\\delta$ of the paired comparisons against "
        "the strongest controlled baseline and against the flat GRU baseline. The HDFS "
        "seed that collapses after epoch 23 is visible in (a) and is why model selection "
        "relies on validation-based early stopping.}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "\\centering",
        "\\includegraphics[width=\\textwidth]{figures/SuppFig2_sensitivity_noise_robustness.pdf}",
        "\\caption*{\\textbf{Supplementary Figure S2.} Hyper-parameter sensitivity and "
        "parsing-noise robustness. (a) Change of validation AUPRC (open circles, selection "
        "criterion) and test F1 (filled squares) relative to the default configuration, in "
        "percentage points. (b) Paired change under template-level parsing noise at a 20\\% "
        "rewrite rate: bars are $\\Delta$F1 (mean, range over two seeds) and open circles "
        "are $\\Delta$AUPRC. Ranking quality is unchanged while F1 moves in both "
        "directions, i.e. the F1 movement reflects threshold calibration rather than a "
        "loss of detection ability.}",
        "\\end{figure}",
    ])

    provenance = pd.read_csv(SUPP / "protocol" / "data_provenance_and_hashes.csv")
    prov_rows = " \\\\\n".join(
        f"{esc(r['dataset'])} & {esc(str(r['cache_built_utc'])[:19])} & "
        f"\\texttt{{{str(r['git_commit'])[:10]}}} & \\texttt{{{str(r['sessions_parquet_sha256'])[:16]}\\ldots}}"
        for _, r in provenance.iterrows())

    document = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=2.3cm]{{geometry}}
\\usepackage{{booktabs,longtable,array,graphicx,amsmath,amssymb,multirow,float}}
\\usepackage{{caption}}
\\captionsetup{{font=small,labelfont=bf,skip=4pt}}
\\usepackage[hidelinks]{{hyperref}}
\\usepackage{{enumitem}}
\\setlist[itemize]{{leftmargin=1.4em,itemsep=2pt,topsep=3pt}}
\\renewcommand{{\\arraystretch}}{{1.05}}
\\sloppy

\\begin{{document}}

\\begin{{center}}
{{\\Large\\bfseries Supplementary Information}}\\\\[6pt]
{{\\large Semantic-Guided Differentiable Hierarchical Temporal Graph Learning\\\\
for Class-Imbalanced Log Anomaly Detection}}\\\\[8pt]
Caizhong Que$^{{1,2}}$, Guofang Dong$^{{1,2,*}}$\\\\[4pt]
{{\\small $^1$ School of Electrical and Information Engineering, Yunnan Minzu University,
Kunming 650504, China\\\\
$^2$ Key Laboratory of Unmanned Autonomous Systems of Yunnan Province,
Yunnan Minzu University, Kunming 650504, China\\\\
$^*$ Corresponding author: dongguofangyx@163.com}}
\\end{{center}}

\\vspace{{-4pt}}
\\hrule
\\vspace{{6pt}}

\\noindent This file contains 13 supplementary tables, two supplementary figures, two
short supplementary texts and the reference list for the literature values quoted in
Supplementary Table S10. Per-seed metrics for all 724 training runs are provided in the
separate workbook \\emph{{Source Data 1}}; the code that produced every item is available
in the accompanying repository.

\\begin{{itemize}}
\\item S1 Statistical testing protocol
\\item S2 Additional metrics for the main comparison
\\item S3 Seed-level results for imbalance learning and prototype configurations
\\item S4 Final hyper-parameters and implementation settings
\\item S5 Hyper-parameter sensitivity analysis
\\item S6 Semantic and entity shortcut ablations
\\item S7 Robustness to parsing noise
\\item S8 Entity identifiability and entity holdout
\\item S9 Masked-template pretraining baseline
\\item S10 Literature reference values and protocol differences
\\item S11 Baseline taxonomy and comparison protocol
\\item S12 Negative-sampling strategies for contrastive pretraining
\\item S13 Semantic field sources and leakage controls
\\item Supplementary Figures S1--S2
\\item Supplementary Text S1 Experimental design and reproducibility protocol
\\item Supplementary Text S2 Data provenance and file hashes
\\end{{itemize}}

\\clearpage
\\subsection*{{Supplementary Text S1. Experimental design and reproducibility protocol}}

\\noindent\\textbf{{Data and leakage control.}} Every dataset is parsed with Drain
(depth 4, similarity 0.5) fitted on training-split events only and then frozen;
validation and test templates that were not seen in training map to a placeholder. The
semantic dictionaries for action and state are fixed before any label is inspected, and
no test-set template, anomaly label, large language model or external anomaly knowledge
base contributes to them. Where a dataset has an official session key (HDFS block,
OpenStack instance, SSH derived key) that key defines the outer sample; otherwise an
adaptive idle-gap threshold, estimated from the training partition alone, is used.
Splits are 60/20/20; BGL and Thunderbird use a session-stratified random split with a
fixed seed, HDFS and OpenStack use a time-ordered split (HDFS groups sessions that share
a raw event to prevent replication leakage), and a time-ordered BGL cache is kept for
the template-evolution stress test.

\\noindent\\textbf{{Model selection and statistics.}} Five seeds (42, 123, 256, 512,
1024) are used for the principal experiments. Models are selected on validation AUPRC
with early stopping (patience 10) and the decision threshold is calibrated on the
validation split by maximising F1; the test split is never used for selection.
Comparative claims are supported by paired Wilcoxon signed-rank tests on matched seeds
with Holm correction inside each comparison family, reported together with Cliff's
$\\delta$; because five paired seeds give a smallest attainable two-sided $p$ of 0.0625,
the paper interprets effect sizes and cross-dataset consistency rather than a binary
significance threshold (Supplementary Table S1).

\\noindent\\textbf{{Reproducibility.}} Each run writes a manifest with the SHA-256
hashes of the raw data, the processed cache and the configuration files, together with
the seed and the effective model configuration; checkpoints store the optimiser state
and the random-number-generator state. The scripts that produce every table and figure
are listed in the repository, and the exact package versions are pinned.

\\clearpage
\\subsection*{{Supplementary Text S2. Data provenance and file hashes}}

\\noindent Processed caches used for the controlled experiments:

\\begin{{center}}
\\begin{{tabular}}{{llll}}
\\toprule
Dataset & Cache built (UTC) & Code revision & sessions.parquet SHA-256 \\\\
\\midrule
{prov_rows} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}

\\noindent The full hashes, the raw-file hashes and the per-run manifests are included
in the repository under \\texttt{{data/processed/*/manifest.json}} and
\\texttt{{outputs/*/main/**/training\\_manifest.json}}.

\\clearpage
{body}

\\clearpage
{figures}

\\clearpage
\\subsection*{{Supplementary References}}
\\begin{{enumerate}}[label={{\\arabic*.}},leftmargin=1.6em,itemsep=1pt]
\\item Guo H, Yuan S, Wu X. LogBERT: log anomaly detection via BERT. In: 2021
International Joint Conference on Neural Networks (IJCNN), 2021.
\\item Du M, Li F, Zheng G, Srikumar V. DeepLog: anomaly detection and diagnosis from
system logs through deep learning. In: Proceedings of the ACM SIGSAC Conference on
Computer and Communications Security, 2017: 1285--1298.
\\item Meng W, Liu Y, Zhu Y, et al. LogAnomaly: unsupervised detection of sequential and
quantitative anomalies in unstructured logs. In: Proceedings of the 28th International
Joint Conference on Artificial Intelligence, 2019: 4739--4745.
\\item Huang S, Liu Y, Fung C, et al. HitAnomaly: hierarchical transformers for anomaly
detection in system log. IEEE Transactions on Network and Service Management, 2020,
17(4): 2064--2076.
\\item Lee Y, Kim J, Kang P. LAnoBERT: system log anomaly detection based on BERT masked
language model. Applied Soft Computing, 2023, 146: 110689.
\\item Zhang C, Peng X, Sha C, et al. DeepTraLog: trace-log combined microservice anomaly
detection through graph-based deep learning. In: Proceedings of the 44th International
Conference on Software Engineering (ICSE), 2022: 623--634.
\\item Liu F, Wen Y, Zhang D, et al. Log2vec: a heterogeneous graph embedding based
approach for detecting cyber threats within enterprise. In: Proceedings of the ACM SIGSAC
Conference on Computer and Communications Security, 2019: 1777--1794.
\\item Ailabouni F, Rom\\'an-Gallego J \\'A, P\\'erez-Delgado M L, Grande P\\'erez L.
Boundary-aware contrastive learning for log anomaly detection. Applied Sciences, 2026,
16(7): 3208.
\\item van Ede T, Aghakhani H, Spahn N, et al. DeepCASE: semi-supervised contextual
analysis of security events. In: Proceedings of the IEEE Symposium on Security and
Privacy, 2022: 522--539.
\\item Hou Z, Ghashami M, Kuznetsov M, Torkamani M. HLogformer: a hierarchical transformer
for representing log data. arXiv preprint arXiv:2408.16803, 2024.
\\item He P, Zhu J, Zheng Z, Lyu M R. Drain: an online log parsing approach with fixed
depth tree. In: Proceedings of the IEEE International Conference on Web Services (ICWS),
2017: 33--40.
\\end{{enumerate}}

\\end{{document}}
"""
    target = SUPP / "Supplementary_Information.tex"
    io.open(target, "w", encoding="utf-8", newline="\n").write(document)
    print("wrote", target)
    print("tables:", len(blocks))


if __name__ == "__main__":
    main()
