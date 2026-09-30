# -*- coding: utf-8 -*-
"""make_supplementary_part2.py -- statistics, hyper-parameters, reference tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PARENT = Path(r"E:\SDHTG\文章\Download+the+journal+article+template+package+(December+2024+version)")
TABLES = PARENT / "supplementary" / "tables"


def write_csv(name, header, rows):
    path = TABLES / name
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("  wrote {} ({} rows)".format(name, len(rows)))


def main():
    # ------------------------------------------------- S1 paired statistics
    report = json.loads(Path("outputs/statistics_report.json").read_text(encoding="utf-8"))
    rows = []
    for row in report["primary"]:
        rows.append(["primary", row["dataset"], row["control_label"], row["metric"],
                     row["delta"] * 100, row["p"], row["q"], row["cliff"], len(row["seeds"])])
    for row in report["baseline"]:
        rows.append(["vs GRU-flat (L0)", row["dataset"], "GRU-flat (L0)", row["metric"],
                     row["delta"] * 100, row["p"], row["q"], row["cliff"], len(row["seeds"])])
    for row in report["modules"]:
        rows.append(["module step", row["dataset"], row["label"], row["metric"],
                     row["delta"] * 100, row["p"], row["q"], row["cliff"], len(row["seeds"])])
    write_csv("S1_paired_statistics.csv",
              ["family", "dataset", "comparison", "metric", "delta_pt", "p_wilcoxon",
               "q_holm", "cliff_delta", "paired_seeds"], rows)

    # ----------------------------------------------------- S4 hyperparameters
    write_csv(
        "S4_hyperparameters.csv",
        ["symbol", "meaning", "value in final model", "sensitivity-scan values",
         "verified against"],
        [
            ["K", "number of normal prototypes", "8", "1, 2, 4, 16",
             "configs/model/sdhtg.yaml: num_normal_prototypes"],
            ["tau_p", "prototype soft-assignment temperature", "0.10", "0.05, 0.20, 0.50",
             "configs/model/sdhtg.yaml: prototype_temperature; sens_prototype_temperature_*"],
            ["lambda_p", "prototype-distance scale", "learned (softplus)",
             "fixed at 0, 0.25, 0.5, 1.0, 2.0",
             "prototype_scale_override defaults to null; sens_prototype_scale_*"],
            ["m", "prototype margin", "0.5", "not scanned (code default family)",
             "configs/experiment/*.yaml: loss.prototype_margin"],
            ["rho_p", "prototype similarity threshold", "0.20 (code default)",
             "not scanned", "src/sdhtg/models/config.py default"],
            ["tau_b", "boundary temperature (annealed 1.0 -> value)", "0.10",
             "final value 0.05, 0.25, 0.50, 1.00",
             "configs/model/sdhtg.yaml: boundary.final_temperature"],
            ["(r_A, r_E)", "target action / entity boundary rates", "0.05 / 0.01",
             "0.01/0.005, 0.01/0.001, 0.10/0.02, 0.10/0.05",
             "configs/experiment/*.yaml: loss.*_boundary_rate"],
            ["R_l", "local temporal radius (status/action/entity)", "8 / 4 / 2",
             "halved (4/2/1), doubled (16/8/4)",
             "configs/model/sdhtg.yaml: local_temporal_radius"],
            ["K_l", "same-semantic neighbours (status/action/entity)", "4 / 4 / 2",
             "halved (2/2/1), doubled (8/8/4)",
             "configs/model/sdhtg.yaml: semantic_neighbors"],
            ["eps_m", "membership-edge threshold", "1e-6", "1e-8, 1e-4, 1e-2",
             "configs/model/sdhtg.yaml: hierarchy.membership_epsilon"],
            ["gamma", "focal modulation exponent", "2.0", "not scanned",
             "configs/experiment/*.yaml: loss.focal_gamma"],
            ["beta", "effective-number weighting exponent", "0.9999", "not scanned",
             "configs/experiment/*.yaml: loss.effective_number_beta"],
            ["w_proto / w_div / w_bal", "prototype, diversity, balance loss weights",
             "0.05 / 0.1 / 0.1", "diversity+balance removed in the final model",
             "configs/experiment/*.yaml: loss.prototype_*_weight"],
            ["--", "masked-template pretraining (LogBERT route, supplementary baseline)",
             "15% mask, 10 epochs", "not scanned",
             "scripts/train_mlm_baseline.py"],
        ],
    )

    # --------------------------------------------- S10 literature reference values
    write_csv(
        "S10_literature_reference_values.csv",
        ["source", "method", "HDFS", "BGL", "Thunderbird", "OpenStack", "SSH",
         "metric", "protocol note"],
        [
            ["original paper", "HitAnomaly", 0.983, 0.921, "", 0.862, "", "F1",
             "fixed time-window sessions; original Figs. 10/16/11"],
            ["original paper", "LogAnomaly", 0.950, 0.960, "", "", "", "F1",
             "sliding window of 20 blocks/sessions"],
            ["original paper", "LogBERT", 0.823, 0.908, 0.966, "", "", "F1",
             "native session sequences; original Table 2"],
            ["original paper", "LAnoBERT", 0.965, 0.875, 0.999, "", "", "F1",
             "parser-free masked LM; predictive-probability score, original Table 3"],
            ["original paper", "DeepCASE", 0.904, "", "", "", "", "F1",
             "event sequences, 80% training data; original Appendix E"],
            ["original paper", "BASN", 0.908, 0.912, "", 0.889, 0.871, "F1",
             "differentiable soft-reset single-level boundaries, zero-shot; original Table 6"],
            ["original paper", "HLogformer", "", "", "", "", "", "F1",
             "evaluated on CloudTrail/OKTA/Amazon Reviews, not on LogHub"],
            ["original paper", "Log2vec", "", "", "", "", "", "AUC",
             "CERT/LANL insider-threat detection, not LogHub"],
            ["original paper", "DeepTraLog", "", "", "", "", "", "F1",
             "TrainTicket microservice traces (F1 0.954), different data type"],
            ["third-party reproduction", "DeepLog", 0.773, 0.861, 0.931, "", "", "F1",
             "same-protocol reproduction reported by the LogBERT paper"],
            ["third-party reproduction", "LogAnomaly", 0.562, 0.741, 0.927, "", "", "F1",
             "same source; 0.29 lower than its original protocol"],
            ["this work (controlled protocol)", "SDHTG", 0.9932, 0.9987, 0.9959,
             0.9991, 0.9565, "F1", "outer samples, five seeds"],
        ],
    )

    # ------------------------------------------------- S11 baseline taxonomy
    write_csv(
        "S11_baseline_taxonomy.csv",
        ["category", "representative methods", "original protocol (input / label unit)",
         "role in this work"],
        [
            ["classical sequence", "DeepLog, LogAnomaly",
             "session template sequences, next-event and count anomalies", "literature reference"],
            ["pretrained language model", "LogBERT, LAnoBERT",
             "masked-template prediction, normal-pattern hypersphere", "literature reference"],
            ["hierarchical", "HitAnomaly, HLogformer",
             "hierarchical Transformer over session sequences or nested log entries",
             "literature reference"],
            ["context construction", "DeepCASE",
             "semi-supervised event clustering, two-stage detection", "literature reference"],
            ["graph-based", "Log2vec, DeepTraLog",
             "representation learning on log graphs or call graphs", "literature reference"],
            ["boundary-aware", "BASN",
             "differentiable soft-reset boundaries, no boundary labels",
             "literature reference (closest related work)"],
            ["controlled baselines", "GRU-flat, TCN, Transformer, GNN-flat",
             "identical template/entity/action/state/time inputs; no learned boundary, "
             "no heterogeneous graph", "implemented here (main comparison)"],
            ["imbalance learning", "BCE, weighted BCE, focal, CB-focal",
             "same inputs and splits, classification loss / prototype setting replaced",
             "implemented here"],
            ["masked-template pretraining", "Transformer + masked-template prediction",
             "identical inputs and encoder, 15% template masking before supervised tuning",
             "implemented here (supplementary baseline)"],
        ],
    )

    # ---------------------------------------------- S13 semantic field provenance
    write_csv(
        "S13_semantic_field_sources.csv",
        ["field", "construction", "source of the labels", "leakage control"],
        [
            ["template", "Drain parser (depth 4, similarity 0.5)",
             "fitted on training-split events only, then frozen",
             "test/validation templates unseen in training map to UNK_TEMPLATE"],
            ["action", "deterministic prior dictionary over training template texts",
             "fixed before seeing any labels",
             "no test-set templates, no LLM, no external anomaly knowledge"],
            ["state", "deterministic status dictionary plus template-identity fallback",
             "fixed before seeing any labels",
             "explicit anomaly words (failed/error/timeout) are ablated in Section 6.4"],
            ["entity", "dataset-specific key (node, block id, host, service)",
             "taken from the raw log fields",
             "train-side vocabulary; unseen entities map to a placeholder"],
            ["time", "per-event inter-arrival gap, continuous-time encoding",
             "derived from the timestamps within an outer sample",
             "no cross-split statistics are used"],
            ["adaptive idle threshold", "median inter-event gap per entity x 2 (capped 3600 s)",
             "estimated on the training partition only",
             "validation/test partitions never re-estimate the threshold"],
        ],
    )
    print("done")


if __name__ == "__main__":
    main()
