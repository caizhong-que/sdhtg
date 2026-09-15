"""
statistical_tests.py -- paired significance testing for the redesigned paper.

Reads per-seed test metrics for a set of method directories and compares every
method against a baseline using paired Wilcoxon signed-rank tests. P-values are
Holm-corrected across comparisons and Cliff's delta is reported as effect size,
as promised by manuscript Section 5.5.

Usage:
    python scripts/statistical_tests.py \
        --root outputs/bgl/main \
        --baseline ladder/L0 \
        --methods ladder/L1 ladder/L2 ladder/L3 ladder/L7 \
        --metric test_auprc

Each method directory must contain seed_*/result.json (with "best_metric" and
"test" keys), as produced by scripts/train.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


METRIC_KEYS = {
    "test_auprc": "auprc",
    "test_auroc": "auroc",
    "test_f1": "f1",
    "test_precision": "precision",
    "test_recall": "recall",
}


def load_per_seed(method_dir: Path, metric: str = "test_auprc") -> dict[int, float]:
    """Load {seed: metric} for one method directory."""
    key = METRIC_KEYS.get(metric, metric)
    values: dict[int, float] = {}
    for seed_dir in sorted(method_dir.glob("seed_*")):
        seed = int(seed_dir.name.replace("seed_", ""))
        result_path = seed_dir / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        test = payload.get("test") or {}
        if key not in test:
            continue
        values[seed] = float(test[key])
    return values


def cliff_delta(control: np.ndarray, treatment: np.ndarray) -> float:
    """Cliff's delta: P(x>y) - P(x<y), sign positive = treatment better."""
    n_c, n_t = control.size, treatment.size
    if n_c == 0 or n_t == 0:
        return float("nan")
    delta = 0.0
    for value in treatment:
        delta += (control < value).sum() - (control > value).sum()
    return float(delta / (n_c * n_t))


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni corrected q-values (manuscript Section 5.5)."""
    order = np.argsort(p_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p_values) + 1)
    m = len(p_values)
    corrected = np.asarray(p_values) * (m - ranks + 1)
    # enforce monotonicity
    for i in range(m - 2, -1, -1):
        corrected[i] = min(corrected[i], corrected[i + 1])
    return corrected.clip(max=1.0).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="experiment output root")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--metric", default="test_auprc")
    args = parser.parse_args()

    if scipy_stats is None:
        raise SystemExit("scipy is required for Wilcoxon tests")

    root = Path(args.root)
    baseline_values = load_per_seed(root / args.baseline, args.metric)
    baseline_keys = sorted(baseline_values)
    baseline_arr = np.asarray([baseline_values[k] for k in baseline_keys])
    print(
        f"baseline {args.baseline}: "
        f"n={len(baseline_arr)} mean={baseline_arr.mean():.4f} "
        f"std={baseline_arr.std(ddof=1):.4f}"
    )
    print(f"{'method':<24}{'n':>4}{'mean':>10}{'std':>10}{'delta':>10}"
          f"{'p':>10}{'q_holm':>10}{'cliff':>10}")

    p_values, names = [], []
    for method in args.methods:
        values = load_per_seed(root / method, args.metric)
        common = [k for k in baseline_keys if k in values]
        if len(common) < 2:
            print(f"{method:<24} insufficient paired seeds")
            continue
        treatment_arr = np.asarray([values[k] for k in common])
        base_common = np.asarray([baseline_values[k] for k in common])
        statistic, p_value = scipy_stats.wilcoxon(
            treatment_arr, base_common, alternative="two-sided"
        )
        p_values.append(float(p_value))
        names.append(method)
        print(
            f"{method:<24}{len(common):>4}{treatment_arr.mean():>10.4f}"
            f"{treatment_arr.std(ddof=1):>10.4f}"
            f"{treatment_arr.mean()-base_common.mean():>10.4f}"
            f"{p_value:>10.4f}{'':>10}{cliff_delta(base_common, treatment_arr):>10.4f}"
        )

    if p_values:
        corrected = holm_correct(p_values)
        print("\nHolm-corrected q-values:")
        for name, q_value in zip(names, corrected):
            print(f"  {name:<24} q={q_value:.4f}")


if __name__ == "__main__":
    main()
