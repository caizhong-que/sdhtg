"""
aggregate_seeds.py

Compute mean +/- standard deviation of ALL test-set metrics across random seeds
for each dataset.  Outputs tables ready for Table 1 (AUPRC, AUROC) and Table 2
(F1, Precision, Recall, MCC, FPR).

Fast derivation path (no GPU needed):
    Reads test_result.json / result.json for precision, recall, samples,
    and quality_report.json for the ground-truth anomaly count, then
    mathematically recovers the confusion matrix to compute MCC and FPR.

Usage:
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --datasets bgl hdfs openstack ssh
    python scripts/aggregate_seeds.py --datasets bgl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

def _recover_confusion(
    precision: float,
    recall: float,
    n_total: int,
    n_positive: int,
) -> dict[str, float]:
    """Recover TP, FP, FN, TN from Precision, Recall, N and P."""
    tp = recall * n_positive
    fp = tp / precision - tp if precision > 0 else 0.0
    fn = n_positive - tp
    tn = n_total - tp - fp - fn
    for k in ("tp", "fp", "fn", "tn"):
        locals()[k] = max(locals()[k], 0.0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def _mcc_fpr(cm: dict[str, float]) -> dict[str, float]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(mcc=mcc, fpr=fpr)


def derive_mcc_fpr(
    precision: float,
    recall: float,
    n_total: int,
    n_positive: int,
) -> dict[str, float]:
    """One-step: confusion matrix -> MCC, FPR."""
    cm = _recover_confusion(precision, recall, n_total, n_positive)
    result = _mcc_fpr(cm)
    result.update(cm)
    return result


# ---------------------------------------------------------------------------
# Per-seed metric load
# ---------------------------------------------------------------------------

def load_seed_metrics(
    seed_dir: Path,
    total_positive: int | None,
) -> dict | None:
    """Load all test metrics for one seed.  Returns None if no data found."""
    tr = None  # the raw test-result dict

    # Prefer test_result.json (list — take last entry)
    tp = seed_dir / "test_result.json"
    if tp.is_file():
        raw = json.loads(tp.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            tr = raw[-1]
        else:
            tr = raw

    # Fall back to result.json -> "test" field
    if tr is None:
        rp = seed_dir / "result.json"
        if rp.is_file():
            rd = json.loads(rp.read_text(encoding="utf-8"))
            tr = rd.get("test")

    if tr is None:
        return None

    # Basic metrics (always present)
    out = {
        "auprc": tr.get("auprc", float("nan")),
        "auroc": tr.get("auroc", float("nan")),
        "f1": tr.get("f1", float("nan")),
        "precision": tr.get("precision", float("nan")),
        "recall": tr.get("recall", float("nan")),
        "samples": tr.get("samples", 0),
    }

    # Derive MCC, FPR, confusion matrix
    prec = out["precision"]
    rec = out["recall"]
    if np.isfinite(prec) and np.isfinite(rec) and total_positive is not None and total_positive > 0:
        derived = derive_mcc_fpr(prec, rec, out["samples"], total_positive)
        out["mcc"] = derived["mcc"]
        out["fpr"] = derived["fpr"]
        out["tp"] = derived["tp"]
        out["fp"] = derived["fp"]
        out["fn"] = derived["fn"]
        out["tn"] = derived["tn"]
    else:
        out["mcc"] = float("nan")
        out["fpr"] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def process_dataset(
    dataset: str,
    output_root: Path,
    experiment: str,
) -> list[dict]:
    """Collect and aggregate metrics across all seeds for one dataset."""
    exp_dir = output_root / dataset / experiment
    if not exp_dir.is_dir():
        print(f"[{dataset}] Not found: {exp_dir}")
        return []

    seed_dirs = sorted(
        [d for d in exp_dir.iterdir()
         if d.is_dir() and d.name.startswith("seed_")],
        key=lambda d: int(d.name.replace("seed_", "")),
    )
    if not seed_dirs:
        print(f"[{dataset}] No seed dirs in {exp_dir}")
        return []

    print(f"\n{'='*72}")
    print(f"[{dataset}] {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")

    # Ground-truth positive count from quality report
    quality_path = Path(f"data/processed/{dataset}/quality_report.json")
    total_positive = None
    if quality_path.is_file():
        qr = json.loads(quality_path.read_text(encoding="utf-8"))
        total_positive = qr.get("splits", {}).get("test", {}).get("anomalous_sessions")
        print(f"  Quality report: test sessions N={qr['splits']['test']['sessions']}, "
              f"anomalous P={total_positive}")
    else:
        print(f"  [WARN] quality_report.json not found, MCC/FPR will be N/A")

    results = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed_", ""))
        m = load_seed_metrics(sd, total_positive)
        if m is None:
            print(f"  seed_{seed}: no test results found --- skip")
            continue
        m["seed"] = seed
        results.append(m)
        print(f"  seed_{seed}: AUPRC={m['auprc']:.4f}  F1={m['f1']:.4f}  "
              f"MCC={m.get('mcc', float('nan')):.4f}  FPR={m.get('fpr', float('nan')):.6f}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METRIC_LABELS = [
    ("auprc", "AUPRC", 4),
    ("auroc", "AUROC", 4),
    ("f1", "F1", 4),
    ("precision", "Precision", 4),
    ("recall", "Recall", 4),
    ("mcc", "MCC", 4),
    ("fpr", "FPR", 6),
]


def fmt_ms(values: list[float], decimals: int) -> str:
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "N/A"
    if len(arr) == 1:
        return f"{arr[0]:.{decimals}f}"
    return f"{arr.mean():.{decimals}f} +/- {arr.std(ddof=1):.{decimals}f}"


def report_metric(name: str, label: str, dec: int, results: list[dict], dataset: str = ""):
    vals = [r.get(name, float("nan")) for r in results]
    return fmt_ms(vals, dec)


def print_tables(all_results: dict[str, list[dict]]):
    """Print two tables: (1) AUPRC+AUROC, (2) F1+P+R+MCC+FPR."""
    # ---- Table 1: AUPRC / AUROC ----
    print(f"\n\n{'='*72}")
    print("TABLE 1 — Ranking metrics (AUPRC, AUROC)")
    print(f"{'='*72}")
    hdr = f"{'Dataset':<14} {'N_seeds':<8} {'AUPRC':<20} {'AUROC':<20}"
    print(hdr)
    print("-" * len(hdr))
    for ds, res in all_results.items():
        if not res:
            print(f"{ds:<14} {'0':<8} {'N/A':<20} {'N/A':<20}")
            continue
        auprc_s = fmt_ms([r["auprc"] for r in res], 4)
        auroc_s = fmt_ms([r["auroc"] for r in res], 4)
        print(f"{ds:<14} {len(res):<8} {auprc_s:<20} {auroc_s:<20}")

    # ---- Table 2: threshold-based metrics ----
    print(f"\n\n{'='*72}")
    print("TABLE 2  — Classification metrics (validation-calibrated threshold)")
    print(f"{'='*72}")
    hdr2 = f"{'Dataset':<12} {'N':<6} {'F1':<18} {'Precision':<18} {'Recall':<18} {'MCC':<18} {'FPR':<18}"
    print(hdr2)
    print("-" * len(hdr2))
    for ds, res in all_results.items():
        if not res:
            print(f"{ds:<12} {'0':<6} {'N/A':<18} {'N/A':<18} {'N/A':<18} {'N/A':<18} {'N/A':<18}")
            continue
        vals = {c: [r.get(c, float("nan")) for r in res] for c in ("f1", "precision", "recall", "mcc", "fpr")}
        f1_s = fmt_ms(vals["f1"], 4)
        p_s = fmt_ms(vals["precision"], 4)
        r_s = fmt_ms(vals["recall"], 4)
        m_s = fmt_ms(vals["mcc"], 4)
        fp_s = fmt_ms(vals["fpr"], 6)
        print(f"{ds:<12} {len(res):<6} {f1_s:<18} {p_s:<18} {r_s:<18} {m_s:<18} {fp_s:<18}")

    # ---- Per-seed detail ----
    print(f"\n\n{'='*72}")
    print("PER-SEED DETAIL")
    print(f"{'='*72}")
    for ds, res in all_results.items():
        if not res:
            continue
        print(f"\n{ds}:")
        print(f"  {'Seed':<6} {'AUPRC':<10} {'AUROC':<10} {'F1':<10} {'P':<10} {'R':<10} {'MCC':<10} {'FPR':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'TN':<8}")
        for r in res:
            print(f"  {r['seed']:<6} "
                  f"{r.get('auprc', float('nan')):<10.4f} "
                  f"{r.get('auroc', float('nan')):<10.4f} "
                  f"{r.get('f1', float('nan')):<10.4f} "
                  f"{r.get('precision', float('nan')):<10.4f} "
                  f"{r.get('recall', float('nan')):<10.4f} "
                  f"{r.get('mcc', float('nan')):<10.4f} "
                  f"{r.get('fpr', float('nan')):<10.6f} "
                  f"{r.get('tp', '?'):<8} "
                  f"{r.get('fp', '?'):<8} "
                  f"{r.get('fn', '?'):<8} "
                  f"{r.get('tn', '?'):<8}")

    # ---- Missing seeds report ----
    print(f"\n\n{'='*72}")
    print("MISSING SEEDS  (no test_result.json / result.json)")
    print(f"{'='*72}")
    any_missing = False
    for ds in all_results:
        exp_dir = Path("outputs") / ds / "main"
        if not exp_dir.is_dir():
            continue
        for sd in sorted(exp_dir.iterdir()):
            if not sd.is_dir() or not sd.name.startswith("seed_"):
                continue
            seed = int(sd.name.replace("seed_", ""))
            found = any(r["seed"] == seed for r in all_results.get(ds, []))
            if not found:
                has_ckpt = (sd / "checkpoints" / "best.pt").is_file()
                has_thresh = (sd / "threshold.json").is_file()
                print(f"  {ds}/{sd.name}: checkpoint={has_ckpt}  threshold={has_thresh}")
                any_missing = True
    if not any_missing:
        print("  (none — all seeds have results)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Aggregate test metrics (mean +/- std) across random seeds"
    )
    p.add_argument("--datasets", nargs="+",
                    default=["bgl", "hdfs", "openstack", "ssh"],
                    help="Dataset names to process")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--experiment", default="main")
    args = p.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    all_results: dict[str, list[dict]] = {}

    for ds in args.datasets:
        all_results[ds] = process_dataset(ds, Path(args.output_root), args.experiment)

    print_tables(all_results)


if __name__ == "__main__":
    main()