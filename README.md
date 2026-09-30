# SDHTG — Semantic-Guided Differentiable Hierarchical Temporal Graph Learning

Reference implementation of **SDHTG**, a log-anomaly-detection model that learns
action-level and entity-level workflow boundaries *inside* a pre-constructed outer
sample, turns the resulting soft membership matrix into both a differentiable
aggregation path (status event → action → entity) and the cross-level edge weights
of a heterogeneous temporal graph, and couples the boundary network to the
detection objective.

The repository contains the full pipeline used for the paper: data preprocessing
with leakage controls, the model and its losses, the incremental ladder protocol,
every baseline and ablation, the interpretation and efficiency studies, and the
scripts that generate the manuscript figures, tables and supplementary material.

## What the paper reports (summary)

| Setting | Outcome |
|---|---|
| HDFS (dynamic range) | F1 **0.9932** vs 0.9530 for the strongest controlled baseline (TCN); AUPRC 0.9974 is slightly below the Transformer (0.9996), i.e. the gain is threshold calibration rather than ranking |
| SSH (dynamic range) | F1 0.9565 vs 0.9515 for the flat GRU baseline; AUPRC 0.9922 vs 0.9854 |
| BGL / OpenStack / Thunderbird | All controlled methods are saturated (F1 ≥ 0.994); SDHTG does not degrade (on Thunderbird a flat GNN is 0.30 pt better, reported as is) |
| Templates unseen (temporal split) | Both methods degrade (flat GRU to AUPRC 0.504); SDHTG keeps a 15.05 pt AUPRC advantage → relative robustness to template evolution |
| Entity identity masked (held-out entities) | BGL F1 drops only 0.33 pt → detection does not rely on entity shortcuts |

Honest reporting is part of the design: negative ablations (prototype
diversity/balance), one late-epoch validation collapse, prototype collapse on
HDFS, and the unfavourable SSH label-scarcity result are all documented in the
paper and reproducible from this code.

## Repository layout

```
src/sdhtg/            model, data pipeline, losses, training loop
  data/               adapters, Drain parsing, semantic priors, sessionization, splits
  models/             SDHTG, baselines (TCN / Transformer / GNN-flat / masked-template)
  losses/             EVL-weighted focal loss, prototype, boundary, contrastive terms
  training/           trainer, curriculum, checkpointing
configs/              experiment / model / data configuration files
scripts/              preprocessing, training, ablations, interpretation, figures, tables
tests/                unit tests for the data pipeline, model and training core
run_*.bat             resumable experiment drivers (Windows)
EXPERIMENT_DESIGN.md  chronological record of the protocol and of every design decision
paper/                manuscript PDF and LaTeX source, supplementary tables, figures, source data
outputs/              generated results (not tracked), one directory per run and seed
data/raw, data/processed   datasets and caches (not tracked)
```

## Environment

Tested with Python 3.11.15, PyTorch 2.5.1 (CUDA 11.8) and scikit-learn 1.5.0 on a
single 8 GB laptop GPU. Install:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements-lock.txt      # exact versions used for the paper
pytest -q                                 # 22 unit tests
```

## Data

All five datasets are public (LogHub / Zenodo): HDFS, BGL, OpenStack, SSH and
Thunderbird. Place the raw files under `data/raw/<DATASET>/` as declared in
`configs/data/<dataset>.yaml` (each `files:` entry lists the expected relative
path), then build a cache:

```bash
python scripts/preprocess.py --config configs/data/bgl.yaml            # session-stratified random split
python scripts/preprocess.py --config configs/data/bgl_temporal.yaml   # time-ordered split (stress test)
```

The parser is fitted on the training partition only, the semantic dictionaries are
fixed before any label is inspected, and unseen test entities and templates map to
placeholders. Every cache is written together with a `manifest.json` that records
the SHA-256 hashes of the raw files and of the products; each training run records
the same hashes in its `training_manifest.json`.

## Quick start

```bash
# 1. full model (L7 configuration) on one dataset and seed
python scripts/train.py --config configs/experiment/ssh.yaml \
    --model-config outputs/_ladder/main/ladder_full/model_L7.yaml \
    --seed 42 --max-epochs 25 --skip-pretrain --tag L7

# 2. incremental ladder L0..L7b on five seeds
python scripts/run_ladder.py --config configs/experiment/ssh.yaml \
    --levels L0,L1,L2,L3,L4,L5,L6,L7,L7b --seeds 42,123,256,512,1024 \
    --max-epochs 25 --pretrain-epochs 0 --tag ladder_full --skip-existing

# 3. controlled baselines (the flat GRU is ladder level L0)
python scripts/run_baselines.py --datasets ssh hdfs --models tcn transformer gnn_flat
```

The experiment drivers (`run_queue.bat`, `run_ablations.bat`, `run_p2.bat`,
`run_parsing_noise.bat`, `run_entity_holdout.bat`, `run_mlm_baseline.bat`) run the
stages strictly sequentially and skip finished runs, so they can be interrupted
and restarted at any time.

## Reproducing the paper

`paper/README.md` maps every manuscript table and figure to the script that
produces it and lists the supplementary tables (S1–S13) with their sources.
Selected entries:

| Paper element | Produced by |
|---|---|
| Table 1 – dataset statistics | `scripts/preprocess.py`, `scripts/audit_summary.py` |
| Table 2, Table 11 – main comparison and efficiency | `scripts/main_comparison.py`, `scripts/benchmark_efficiency_v2.py` |
| Table 5 – incremental ablation | `scripts/run_ladder.py`, `scripts/analyze_ladder.py` |
| Table 8 – template-evolution stress test | `scripts/run_ladder.py --config configs/experiment/bgl_temporal.yaml` |
| Figures 1–7 | `scripts/plot_fig_architecture.py`, `plot_fig_main_results.py`, `plot_fig_pr_threshold.py`, `plot_fig_structure_evidence.py`, `plot_fig_interpretability.py`, `plot_case_figure.py`, `plot_fig_efficiency.py` |
| Supplementary S1 – paired statistics | `scripts/statistics_report.py` |
| Supplementary S5 – sensitivity scan | `scripts/run_sensitivity.py` |
| Supplementary S7 – parsing noise | `scripts/make_noisy_dataset.py`, `scripts/run_parsing_noise.py`, `scripts/evaluate_noise_paired.py` |
| Supplementary S8 – entity identifiability | `scripts/run_entity_holdout.py`, `scripts/evaluate_seen_unseen.py` |
| Supplementary S9 – masked-template baseline | `scripts/train_mlm_baseline.py` |
| Supplementary package build | `scripts/make_supplementary.py`, `scripts/make_supplementary_part2.py` |

Five seeds (42, 123, 256, 512, 1024) are used for the principal experiments.
Because only five paired seeds exist, the paper reports effect sizes (Cliff's δ)
and Holm-corrected Wilcoxon q values instead of claiming a binary significance
threshold; the minimum attainable two-sided p with five pairs is 0.0625.

## Paper artifacts

`paper/manuscript/` holds the submission PDF and its LaTeX/BibTeX source (the
Springer Nature class file is not redistributed; download the template package
from the publisher to compile). `paper/supplementary/` holds Supplementary Tables
S1–S13, Supplementary Figures S1–S2, the per-seed source data for all 724 runs and
the data-provenance table.

## Known issues and notes

These are recorded openly because they affect interpretation or further work:

1. **SSH cache split.** The cache under `data/processed/ssh` was built before the
   current split code was committed; its actual partition sizes (77.8 % / 11.1 % /
   11.1 %) are not time ordered, unlike what the configuration implies.
   Regenerate the cache with `scripts/preprocess.py --config configs/data/ssh.yaml`
   and re-run the SSH experiments before relying on that protocol statement.
2. **Cross-level messages belong to the final model.** `use_cross_level_messages`
   is `true` in both `configs/model/sdhtg.yaml` and the ladder's L7 configuration
   (HDFS ΔF1 = +2.17 pt). Only the prototype diversity/balance term (L7b) is
   removed by the retention rule.
3. **`K_m` is not implemented.** It appears in an early draft of the
   hyper-parameter table; the code has no such key. Supplementary Table S4 lists
   the verified parameter set.
4. The per-stage cost attribution in the efficiency figure comes from
   `scripts/diagnose_step_cost.py`; re-run it to regenerate the profiling record.
5. Label scarcity (BGL: 2 seeds, SSH: 3 seeds) and parsing noise (SSH: 2 seeds)
   use fewer seeds than the principal experiments; the paper and the supplementary
   tables state the counts explicitly.

## Citation

```bibtex
@article{sdhtg2026,
  title  = {Semantic-Guided Differentiable Hierarchical Temporal Graph Learning
            for Class-Imbalanced Log Anomaly Detection},
  author = {Que, Caizhong and Dong, Guofang},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

MIT (see `LICENSE`). The datasets remain under their original LogHub terms and are
not redistributed here.
