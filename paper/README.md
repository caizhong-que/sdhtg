# Paper artifacts and reproduction map

## Contents

```
manuscript/SDHTG_manuscript.pdf     submission PDF (Springer Nature template)
manuscript/sn-article.tex           LaTeX source (requires the publisher's sn-jnl.cls)
manuscript/sn-bibliography.bib      bibliography
supplementary/Supplementary_Information.pdf   single SI PDF (cover page, Text S1-S2,
                                              Tables S1-S13, Figures S1-S2, references)
supplementary/Supplementary_Information.tex   LaTeX source of the SI
supplementary/Source_Data_1.xlsx    per-seed source data (11 worksheets, all 724 runs)
supplementary/tables/               Supplementary Tables S1-S13 (CSV, UTF-8 BOM)
supplementary/figures/              Supplementary Figures S1-S2 (PDF/PNG/SVG)
supplementary/source_data/          per-seed metrics for all 724 runs, statistics report,
                                    efficiency benchmark records
supplementary/protocol/             experiment design and data-provenance hashes
supplementary/Supplementary_information_statement.md   text to paste into the manuscript
supplementary/README_索引与提交说明.md                 index and submission notes (Chinese)
```

## Building the supplementary PDF

```bash
python scripts/make_supplementary.py            # tables S1-S13 as CSV + source data
python scripts/make_supplementary_part2.py      # statistics, hyper-parameters, reference tables
python scripts/make_supplementary_pdf.py        # Supplementary_Information.tex
python scripts/make_source_data_workbook.py     # Source_Data_1.xlsx
cd ../文章/"Download+the+journal+article+template+package+(December+2024+version)"/supplementary
xelatex -interaction=nonstopmode Supplementary_Information.tex   # run twice
```

## Element-to-script map

### Main tables

| Element | Scripts / commands | Data source |
|---|---|---|
| Table 1 — dataset statistics | `scripts/preprocess.py`, `scripts/audit_summary.py` | `data/processed/*/quality_report.json` |
| Table 2 — main comparison | `scripts/main_comparison.py` (or `scripts/make_paper_tables.py`) | `outputs/<dataset>/main/{baseline_*,ladder_full/L0,ladder_full/L7}/seed_*/result.json` |
| Table 3 — boundary localization | `scripts/make_synthetic_structural.py`, `scripts/evaluate_boundary.py`, `scripts/evaluate_boundary_baselines.py` | `outputs/bgl_synth/**` |
| Table 4 — evidence faithfulness | `scripts/evaluate_faithfulness.py` | 60 semi-synthetic test samples |
| Table 5 — incremental ladder | `scripts/run_ladder.py`, `scripts/analyze_ladder.py` | `outputs/<dataset>/main/ladder_full/L*/seed_*/result.json` |
| Table 6 — imbalance / prototype | `scripts/run_ablations.py`, `scripts/summarize_ablations.py` | `outputs/ssh/main/ablation_imbalance_*/seed_*/result.json` |
| Table 7 — label scarcity | `scripts/run_label_scarcity.py`, `scripts/analyze_label_scarcity.py` | `outputs/{bgl,ssh}/main/label_scarcity/**` |
| Table 8 — template evolution | `scripts/run_ladder.py --config configs/experiment/bgl_temporal.yaml` | `outputs/bgl_temporal/main/temporal/**` |
| Table 9 — boundary evidence vs priors | `scripts/interpret_boundaries.py` | `outputs/<dataset>/main/ladder_full/L7/interpretability/**` |
| Tables 10–11 — efficiency | `scripts/benchmark_efficiency_v2.py`, `scripts/show_efficiency.py` | `outputs/efficiency_benchmark.json` |

### Main figures

| Element | Script |
|---|---|
| Fig. 1 architecture | `scripts/plot_fig_architecture.py` |
| Fig. 2 main comparison | `scripts/plot_fig_main_results.py` |
| Fig. 3 ranking vs threshold | `scripts/plot_fig_pr_threshold.py` |
| Fig. 4 structural identifiability / faithfulness | `scripts/plot_fig_structure_evidence.py` |
| Fig. 5 hierarchical evidence | `scripts/plot_fig_interpretability.py` |
| Fig. 6 HDFS cases | `scripts/plot_case_figure.py` |
| Fig. 7 efficiency | `scripts/plot_fig_efficiency.py` |

All plotting scripts use `scripts/_figure_common.py`, which provides the shared
style plus four layout self-checks (canvas overflow, legend/data overlap,
text/text overlap, inked border).

### Supplementary tables (S1–S13)

| Table | Content | Source |
|---|---|---|
| S1 | paired statistics (Wilcoxon, Holm q, Cliff's δ) | `scripts/statistics_report.py` |
| S2 | complete secondary metrics of the main comparison | `scripts/make_supplementary.py` (derives MCC and balanced accuracy from precision/recall and the test composition) |
| S3 | seed-level imbalance/prototype results | `scripts/make_supplementary.py` |
| S4 | verified hyper-parameter inventory | `scripts/make_supplementary_part2.py`, checked against `configs/` and `scripts/audit_final_config.py` |
| S5 | sensitivity scan (78 runs) | `scripts/run_sensitivity.py`, aggregated by `scripts/make_supplementary.py` |
| S6 | semantic/entity shortcut ablations | `scripts/run_ablations.py` |
| S7 | parsing noise at 20 % and 40 % rewrite | `scripts/make_noisy_dataset.py`, `scripts/run_parsing_noise.py`, `scripts/evaluate_noise_paired.py` |
| S8 | entity-identifiability contrast | `scripts/make_entity_holdout.py`, `scripts/run_entity_holdout.py`, `scripts/evaluate_seen_unseen.py` |
| S9 | masked-template pretraining baseline | `scripts/train_mlm_baseline.py` (model in `src/sdhtg/models/baselines.py`) |
| S10 | literature-reported reference values | curated from the original papers (verified against the PDFs) |
| S11 | baseline taxonomy | curated |
| S12 | negative-sampling pretraining study | `scripts/run_pretraining.py` |
| S13 | semantic-field provenance | curated from `src/sdhtg/data/semantic_prior.py` and the preprocessing protocol |
| S1–S2 figures | training dynamics; sensitivity and parsing noise | `scripts/plot_fig_training_effect.py`, `scripts/plot_fig_sensitivity_robustness.py` |

### Auxiliary audit scripts

`scripts/audit_final_config.py` (split protocol, ablation flags, prototype
configuration), `scripts/audit_cache_provenance.py` (cache build time and code
version), `scripts/audit_table_inventory.py` (table inventory, size and citation
count), `scripts/check_math_shifts.py` (LaTeX math-delimiter lint) and
`scripts/audit_final_manuscript.py` (inventory of the final manuscript).
