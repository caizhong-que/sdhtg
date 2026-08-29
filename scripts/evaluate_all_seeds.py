"""
evaluate_all_seeds.py

Load best.pt and threshold.json for every seed of each dataset, run test
inference (GPU required), compute AUPRC, AUROC, F1, Precision, Recall, MCC,
FPR using the validation-calibrated threshold, then aggregate across seeds
(mean +/- std).

Usage:
    python scripts/evaluate_all_seeds.py --datasets bgl hdfs openstack ssh
    python scripts/evaluate_all_seeds.py --datasets ssh               # single dataset
    python scripts/evaluate_all_seeds.py --save                        # save per-seed + aggregated JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.data.shortcuts import (
    mask_entity_to_unk,
    mask_explicit_status_words,
    shuffle_entity_ids,
    status_word_mask_from_vocab,
)
from sdhtg.models.factory import build_model
from sdhtg.training.reproducibility import seed_everything

# ---------------------------------------------------------------------------
# BucketBatchSampler (identical to evaluate.py)
# ---------------------------------------------------------------------------

class BucketBatchSampler(torch.utils.data.Sampler):
    BUCKET_KEYS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]

    def __init__(self, lengths, batch_size, shuffle=False):
        buckets = {}
        for idx, l in enumerate(lengths):
            for b in self.BUCKET_KEYS:
                if l <= b:
                    buckets.setdefault(b, []).append(idx)
                    break
        self.batches = []
        for _, indices in sorted(buckets.items()):
            key = _
            if key <= 4:
                bs = max(512, batch_size)
            elif key <= 8:
                bs = min(512, batch_size // 4)
            elif key <= 16:
                bs = min(256, batch_size // 8)
            elif key <= 64:
                bs = 128
            else:
                bs = 64
            bs = min(bs, len(indices))
            for i in range(0, len(indices), bs):
                self.batches.append(indices[i:i + bs])

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


# ---------------------------------------------------------------------------
# Single-seed evaluation
# ---------------------------------------------------------------------------

def evaluate_seed(
    checkpoint_path: Path,
    threshold_path: Path,
    config_path: Path,
    seed: int,
    device: torch.device,
    batch_size: int = 2048,
    processed_dir: str | None = None,
    model_config: str | None = None,
    entity_to_unk: bool = False,
    shuffle_entity_id: bool = False,
    mask_status_words: bool = False,
) -> dict:
    """Run test inference for one seed and compute all metrics using the
    validation-calibrated threshold from threshold.json."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if processed_dir is not None:
        cfg["data"]["processed_dir"] = processed_dir
    seed_everything(seed, bool(cfg.get("deterministic", True)))

    processed = Path(cfg["data"]["processed_dir"])
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    status_word_mask = status_word_mask_from_vocab(vocab["status"])
    overrides = {f"{n}_vocab_size": len(vocab[n]) for n in ["template", "entity", "action", "status"]}

    # Build model
    model = build_model(
        model_config or cfg["model_config"], overrides
    ).to(device).eval()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    if any(k.startswith("model.") for k in sd):
        sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    epoch = ckpt.get("epoch", "?")

    # Load validation-calibrated threshold
    th_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = th_data["threshold"]

    # Test data loader
    ds = SessionDataset(str(processed / "sessions.parquet"), "test")
    sampler = BucketBatchSampler(ds.lengths, batch_size)
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate_sessions,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
    )

    # Inference
    # use_amp = device.type == "cuda"
    use_amp = False
    y_true_list, y_score_list = [], []
    for batch in tqdm(loader, desc=f"seed_{seed}", unit="batch", leave=False):
        batch = move_batch_to_device(batch, device)
        if entity_to_unk:
            mask_entity_to_unk(batch)
        if shuffle_entity_id:
            shuffle_entity_ids(batch, len(vocab["entity"]), seed=0)
        if mask_status_words:
            mask_explicit_status_words(batch, status_word_mask)
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch)
            else:
                out = model(batch)
        y_true_list.extend(batch["label"].cpu().tolist())
        y_score_list.extend(out.anomaly_probability.float().cpu().tolist())

    y_true = np.asarray(y_true_list, dtype=int)
    y_score = np.asarray(y_score_list, dtype=float)

    # Ranking metrics (threshold-independent)
    auprc = float(average_precision_score(y_true, y_score))
    n_classes = len(np.unique(y_true))
    auroc = float(roc_auc_score(y_true, y_score)) if n_classes > 1 else float("nan")

    # Apply validation-calibrated threshold
    y_pred = (y_score >= threshold).astype(int)

    # Classification metrics
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Confusion matrix components
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fpr_val = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "seed": seed,
        "epoch": epoch,
        "auprc": auprc,
        "auroc": auroc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "mcc": mcc,
        "fpr": fpr_val,
        "threshold_applied": threshold,
        "samples": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "checkpoint": str(checkpoint_path),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = ["auprc", "auroc", "f1", "precision", "recall", "mcc", "fpr"]


def _mean_std(values: list[float], decimals: int = 4) -> str:
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "N/A"
    if len(arr) == 1:
        return f"{arr[0]:.{decimals}f}"
    return f"{arr.mean():.{decimals}f} +/- {arr.std(ddof=1):.{decimals}f}"


def aggregate_seeds(results: list[dict]) -> dict:
    """Compute mean, std, min, max, n for each metric across seeds."""
    agg = {"n_seeds": len(results)}
    for k in METRIC_KEYS:
        vals = [r[k] for r in results if np.isfinite(r.get(k, float("nan")))]
        if not vals:
            agg[k] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
            continue
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }
    return agg


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    dataset: str,
    output_root: Path,
    experiment: str = "main",
    device: torch.device = torch.device("cpu"),
    batch_size: int = 2048,
    save: bool = False,
    only_missing: bool = False,
    entity_to_unk: bool = False,
    shuffle_entity_id: bool = False,
    mask_status_words: bool = False,
) -> list[dict]:
    """Evaluate all seeds for one dataset and aggregate."""
    exp_dir = output_root / dataset / experiment
    if not exp_dir.is_dir():
        print(f"[{dataset}] Not found: {exp_dir}")
        return []

    # Prefer each seed's training manifest; fall back to the dataset data config.
    data_cfg_path = Path(f"configs/data/{dataset}.yaml")
    fallback_processed = None
    if data_cfg_path.is_file():
        fallback_processed = yaml.safe_load(
            data_cfg_path.read_text(encoding="utf-8")
        ).get("processed_dir")

    seed_dirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")],
        key=lambda d: int(d.name.replace("seed_", "")),
    )
    if not seed_dirs:
        print(f"[{dataset}] No seed dirs in {exp_dir}")
        return []

    config_path = Path(f"configs/experiment/{experiment}.yaml")
    if not config_path.is_file():
        fallback = Path("configs/experiment/main.yaml")
        if not fallback.is_file():
            print(f"[{dataset}] Config not found: {config_path}")
            return []
        print(f"[{dataset}] Experiment config missing; using {fallback}")
        config_path = fallback

    print(f"\n{'=' * 72}")
    print(f"[{dataset}] {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")
    print(f"  Config: {config_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    if only_missing:
        print(f"  Only missing seeds (no test_result.json)")

    results = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed_", ""))
        ckpt_path = sd / "checkpoints" / "best.pt"
        thresh_path = sd / "threshold.json"

        if not ckpt_path.is_file():
            print(f"  seed_{seed}: no best.pt --- skip")
            continue
        if not thresh_path.is_file():
            print(f"  seed_{seed}: no threshold.json --- skip")
            continue

        if only_missing:
            # Skip if test_result.json already exists
            if (sd / "test_result.json").is_file():
                print(f"  seed_{seed}: already has test_result.json --- skip")
                continue

        t0 = time.perf_counter()
        manifest_path = sd / "training_manifest.json"
        seed_processed = None
        seed_model_config = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for data_path in manifest.get("data", {}):
                if "sessions.parquet" in data_path:
                    seed_processed = str(Path(data_path).parent)
                    break
            for config_path_key in manifest.get("configs", {}):
                if "model" in config_path_key and config_path_key.endswith(".yaml"):
                    seed_model_config = config_path_key
                    break
        if seed_processed is None:
            seed_processed = fallback_processed
        result = evaluate_seed(
            ckpt_path, thresh_path, config_path, seed, device, batch_size,
            processed_dir=seed_processed,
            model_config=seed_model_config,
            entity_to_unk=entity_to_unk,
            shuffle_entity_id=shuffle_entity_id,
            mask_status_words=mask_status_words,
        )
        elapsed = time.perf_counter() - t0
        results.append(result)

        print(f"  seed_{seed}: AUPRC={result['auprc']:.4f}  AUROC={result['auroc']:.4f}  "
              f"F1={result['f1']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}  "
              f"MCC={result['mcc']:.4f}  FPR={result['fpr']:.6f}  ({elapsed:.0f}s)")

    if not results:
        print(f"  No seeds evaluated.")
        return []

    # Aggregate
    agg = aggregate_seeds(results)
    print(f"\n  >>> Aggregation ({len(results)} seeds):")
    for k in METRIC_KEYS:
        print(f"      {k.upper():<12} {_mean_std([r[k] for r in results], 6 if k == 'fpr' else 4)}")
    print(f"      {'SAMPLES':<12} {results[0]['samples']:<12}")

    # Save
    if save:
        out = {
            "dataset": dataset,
            "experiment": experiment,
            "config": str(config_path),
            "n_seeds_found": len(seed_dirs),
            "n_seeds_evaluated": len(results),
            "aggregated": agg,
            "per_seed": results,
        }
        save_path = exp_dir / "aggregated_metrics.json"
        save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  Saved to: {save_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate all seeds: load best.pt, run test inference, "
                    "compute AUPRC/AUROC/F1/P/R/MCC/FPR, aggregate mean +/- std"
    )
    p.add_argument("--datasets", nargs="+", default=["bgl", "hdfs", "openstack", "ssh"],
                   help="Dataset names")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--experiment", default="main")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--save", action="store_true", help="Save per-seed + aggregated results to JSON")
    p.add_argument("--only-missing", action="store_true",
                   help="Only evaluate seeds without test_result.json (useful for filling gaps)")
    p.add_argument("--entity-unk", action="store_true",
                   help="Map every entity ID to UNK during evaluation")
    p.add_argument("--shuffle-entity-id", action="store_true",
                   help="Shuffle entity IDs (fixed seed) during evaluation")
    p.add_argument("--mask-status-words", action="store_true",
                   help="Map explicit anomaly status words to UNK during evaluation")
    args = p.parse_args()

    sys.path.insert(0, str(Path.cwd()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available — inference will be very slow on CPU!")

    all_results = {}
    for ds in args.datasets:
        all_results[ds] = process_dataset(
            ds, Path(args.output_root), args.experiment,
            device, args.batch_size, args.save, args.only_missing,
            args.entity_unk, args.shuffle_entity_id, args.mask_status_words,
        )

    # Final aggregation table
    print(f"\n\n{'=' * 72}")
    print("FINAL AGGREGATION — All metrics across all datasets")
    print(f"{'=' * 72}")
    hdr = (f"  {'Dataset':<12} {'N':<5} {'AUPRC':<18} {'AUROC':<18} {'F1':<18} "
           f"{'Precision':<18} {'Recall':<18} {'MCC':<18} {'FPR':<18}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for ds, res in all_results.items():
        if not res:
            print(f"  {ds:<12} {'0':<5} {'N/A':<18} {'N/A':<18} {'N/A':<18} "
                  f"{'N/A':<18} {'N/A':<18} {'N/A':<18} {'N/A':<18}")
            continue
        vals = {k: [r[k] for r in res] for k in METRIC_KEYS}
        decs = {"auprc": 4, "auroc": 4, "f1": 4, "precision": 4, "recall": 4, "mcc": 4, "fpr": 6}
        parts = [f"{ds:<12}", f"{len(res):<5}"]
        for k in METRIC_KEYS:
            parts.append(_mean_std(vals[k], decs[k]).ljust(18))
        print("  " + " ".join(parts))


if __name__ == "__main__":
    main()
