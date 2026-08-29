"""
Orchestrate logbert baselines (DeepLog, LogAnomaly, LogBERT) across
5 datasets (bgl, hdfs, openstack, ssh, thunderbird) and 5 seeds (42, 123, 256, 512, 1024).

Outputs all 7 metrics: AUPRC, AUROC, F1, Precision, Recall, MCC, FPR.

Usage:
    # Full run (all datasets, all baselines, all seeds)
    python scripts/run_logbert_baselines.py --mode all

    # Step-by-step
    python scripts/run_logbert_baselines.py --mode data_process --datasets bgl hdfs
    python scripts/run_logbert_baselines.py --mode vocab --baselines deeplog logbert
    python scripts/run_logbert_baselines.py --mode train --seeds 42 123
    python scripts/run_logbert_baselines.py --mode predict --seeds 42 123
    python scripts/run_logbert_baselines.py --mode aggregate

    # Quick test (single combo)
    python scripts/run_logbert_baselines.py --mode all --datasets bgl --baselines deeplog --seeds 42
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

SEEDS = [42, 123, 256, 512, 1024]
DATASETS = {
    "bgl":        "BGL",
    "hdfs":       "HDFS",
    "openstack":  "OpenStack",
    "ssh":        "SSH",
    "thunderbird":"TBird",
}
BASELINES = ["deeplog", "loganomaly", "logbert"]

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baselines" / "logbert"
METRIC_KEYS = ["auprc", "auroc", "f1", "precision", "recall", "mcc", "fpr"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _dataset_dir(dataset: str) -> Path:
    """Get the logbert directory for a dataset."""
    dir_name = DATASETS[dataset]
    return BASELINE_DIR / dir_name


def _output_dir(dataset: str) -> Path:
    """Get the output directory for a dataset."""
    return BASELINE_DIR.parent.parent / "outputs" / dataset


def _save_dir(dataset: str, baseline: str) -> Path:
    """Get the save directory for a dataset+baseline."""
    p = _output_dir(dataset) / baseline
    return p


def _run(cmd: list[str], cwd: Path, desc: str = ""):
    """Run a command and return success."""
    print(f"  [{desc}] {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):")
        for line in result.stderr.splitlines()[-10:]:
            print(f"    {line}")
        return False
    # Print last 5 lines of stdout for progress info
    for line in result.stdout.splitlines()[-5:]:
        if any(k in line.lower() for k in ["auprc", "auroc", "mcc", "fpr", "f1", "precision", "recall",
                                           "tp:", "fp:", "fn:", "tn:", "error", "done"]):
            print(f"    {line}")
    return True


def _fmt_mean_std(values: list[float], decimals: int = 4) -> str:
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "N/A"
    if len(arr) == 1:
        return f"{arr[0]:.{decimals}f}"
    return f"{arr.mean():.{decimals}f} +/- {arr.std(ddof=1):.{decimals}f}"


# ------------------------------------------------------------
# Steps
# ------------------------------------------------------------

def step_data_process(datasets: list[str]):
    """Step 1: Run data_process.py for each dataset."""
    for ds in datasets:
        d = _dataset_dir(ds)
        script = d / "data_process.py"
        if not script.exists():
            print(f"[{ds}] data_process.py not found in {d}")
            continue
        print(f"\n{'='*60}")
        print(f"  [{ds}] Running data_process.py")
        print(f"{'='*60}")
        _run([sys.executable, "data_process.py"], d, f"{ds}_data_process")


def step_vocab(datasets: list[str], baselines: list[str]):
    """Step 2: Build vocabulary for each baseline."""
    for ds in datasets:
        d = _dataset_dir(ds)
        for bl in baselines:
            script = d / f"{bl}.py"
            if not script.exists():
                print(f"[{ds}/{bl}] {script} not found, skip")
                continue
            print(f"  [{ds}/{bl}] Building vocab...")
            _run([sys.executable, f"{bl}.py", "vocab"], d, f"{ds}_{bl}_vocab")


def step_train(datasets: list[str], baselines: list[str], seeds: list[int]):
    """Step 3: Train model for each dataset/baseline/seed."""
    for ds in datasets:
        d = _dataset_dir(ds)
        for bl in baselines:
            script = d / f"{bl}.py"
            if not script.exists():
                continue
            for seed in seeds:
                print(f"\n  [{ds}/{bl}] Training seed={seed}...")
                _run([sys.executable, f"{bl}.py", "train", "--seed", str(seed)], d, f"{ds}_{bl}_seed{seed}")


def step_predict(datasets: list[str], baselines: list[str], seeds: list[int]):
    """Step 4: Run prediction (computes all 7 metrics)."""
    for ds in datasets:
        d = _dataset_dir(ds)
        for bl in baselines:
            script = d / f"{bl}.py"
            if not script.exists():
                continue
            for seed in seeds:
                print(f"\n  [{ds}/{bl}] Predicting seed={seed}...")
                _run([sys.executable, f"{bl}.py", "predict"], d, f"{ds}_{bl}_seed{seed}_predict")


def step_aggregate(datasets: list[str], baselines: list[str], seeds: list[int]):
    """Step 5: Aggregate test results across seeds, compute metrics summary."""
    print(f"\n\n{'='*72}")
    print("FINAL AGGREGATION — All baselines × datasets × seeds")
    print(f"{'='*72}")

    for bl in baselines:
        print(f"\n\n--- Baseline: {bl.upper()} ---")
        header = f"  {'Dataset':<12} {'N':<5}"
        for k in METRIC_KEYS:
            header += f" {k.upper():<16}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for ds in datasets:
            save_dir = _save_dir(ds, bl)
            results = []
            for seed in seeds:
                res_file = save_dir / f"test_result_seed{seed}.json"
                if not res_file.exists():
                    continue
                try:
                    data = json.loads(res_file.read_text(encoding="utf-8"))
                    results.append(data)
                except Exception:
                    pass

            if not results:
                print(f"  {ds:<12} {'0':<5} {'N/A':<16} ...")
                continue

            parts = [f"{ds:<12}", f"{len(results):<5}"]
            for k in METRIC_KEYS:
                vals = [r.get(k, float("nan")) for r in results]
                parts.append(f"{_fmt_mean_std(vals, 6 if k == 'fpr' else 4):<16}")
            print("  " + " ".join(parts))

    # Per-seed detail
    print(f"\n\n{'='*72}")
    print("PER-SEED DETAIL")
    print(f"{'='*72}")
    for bl in baselines:
        print(f"\n  Baseline: {bl.upper()}")
        for ds in datasets:
            save_dir = _save_dir(ds, bl)
            detail = []
            for seed in seeds:
                res_file = save_dir / f"test_result_seed{seed}.json"
                if res_file.exists():
                    try:
                        data = json.loads(res_file.read_text(encoding="utf-8"))
                        detail.append((seed, data))
                    except Exception:
                        pass
            if not detail:
                continue
            print(f"  [{ds}]")
            h = f"    {'Seed':<6}"
            for k in METRIC_KEYS:
                h += f" {k.upper():<12}"
            print(h)
            for seed, data in detail:
                row = f"    {seed:<6}"
                for k in METRIC_KEYS:
                    v = data.get(k, float("nan"))
                    row += f" {v:<12.4f}" if isinstance(v, (int, float)) else f" {str(v):<12}"
                print(row)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run logbert baselines evaluation")
    p.add_argument("--mode", default="aggregate",
                    choices=["data_process", "vocab", "train", "predict", "aggregate", "all"],
                    help="Pipeline step to run")
    p.add_argument("--datasets", nargs="+",
                    default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()),
                    help="Datasets to process")
    p.add_argument("--baselines", nargs="+",
                    default=BASELINES,
                    choices=BASELINES,
                    help="Baselines to run")
    p.add_argument("--seeds", nargs="+", type=int,
                    default=SEEDS,
                    help="Random seeds")
    args = p.parse_args()

    mode = args.mode
    datasets = args.datasets
    baselines = args.baselines
    seeds = args.seeds

    print(f"Mode: {mode}")
    print(f"Datasets: {datasets}")
    print(f"Baselines: {baselines}")
    print(f"Seeds: {seeds}")

    if mode in ("data_process", "all"):
        step_data_process(datasets)

    if mode in ("vocab", "all"):
        step_vocab(datasets, baselines)

    if mode in ("train", "all"):
        step_train(datasets, baselines, seeds)

    if mode in ("predict", "all"):
        step_predict(datasets, baselines, seeds)

    if mode in ("aggregate", "all"):
        step_aggregate(datasets, baselines, seeds)

    print("\nDone!")


if __name__ == "__main__":
    main()
