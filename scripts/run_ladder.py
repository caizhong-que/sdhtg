"""
run_ladder.py -- incremental component validation for SDHTG.

The model is grown one module at a time. Every level is a strict superset of
the previous one, and each level is validated with the same seeds/epochs
before the next module is admitted. This implements the "validate before you
add" protocol used for the redesigned experiment section.

Usage:
    python scripts/run_ladder.py --config configs/experiment/ssh.yaml \
        --levels L0,L1,L2,L3,L4,L5,L6,L7,L8 --seeds 42,123 \
        --max-epochs 25 --pretrain-epochs 10 --tag ladder_ssh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Level definitions. Each level enables exactly one additional component on
# top of the previous level. The final level (L8) is the full model with
# contrastive pretraining; it uses the same architecture as L7.
# ---------------------------------------------------------------------------

# name -> ablation flags that are ON at this level. Each level is a strict
# SUPERSET of the previous one (flags accumulate), so L3+ keeps the hierarchy
# enabled by L2, L4+ keeps the entity boundary enabled by L3, and so on.
LEVEL_ABLATIONS = {
    "L0": {},  # flat GRU + CB-Focal only
    "L1": {"use_strategy_film": True},  # + contextual FiLM
    "L2": {  # + learned action boundary and hierarchical aggregation
        "use_hierarchy": True,
        "use_action_boundary": True,
    },
    "L3": {  # + nested entity boundary
        "use_hierarchy": True,
        "use_action_boundary": True,
        "use_entity_boundary": True,
    },
    "L4": {  # + local temporal edges
        "use_hierarchy": True,
        "use_action_boundary": True,
        "use_entity_boundary": True,
        "use_temporal_edges": True,
    },
    "L5": {  # + same-semantic edges
        "use_hierarchy": True,
        "use_action_boundary": True,
        "use_entity_boundary": True,
        "use_temporal_edges": True,
        "use_semantic_edges": True,
    },
    "L6": {  # + cross-level containment edges
        "use_hierarchy": True,
        "use_action_boundary": True,
        "use_entity_boundary": True,
        "use_temporal_edges": True,
        "use_semantic_edges": True,
        "use_cross_level_messages": True,
    },
    "L7": {  # + multi-normal prototypes (lambda_p, delta_p)
        "use_hierarchy": True,
        "use_action_boundary": True,
        "use_entity_boundary": True,
        "use_temporal_edges": True,
        "use_semantic_edges": True,
        "use_cross_level_messages": True,
        "use_prototypes": True,
    },
    "L8": {  # L7 + contrastive pretraining (handled separately)
        "use_hierarchy": True,
        "use_action_boundary": True,
        "use_entity_boundary": True,
        "use_temporal_edges": True,
        "use_semantic_edges": True,
        "use_cross_level_messages": True,
        "use_prototypes": True,
    },
    "L7b": {  # L7 + differentiable prototype diversity/balance losses
        "use_hierarchy": True,
        "use_action_boundary": True,
        "use_entity_boundary": True,
        "use_temporal_edges": True,
        "use_semantic_edges": True,
        "use_cross_level_messages": True,
        "use_prototypes": True,
    },
}

LEVEL_LABELS = {
    "L0": "基础 GRU",
    "L1": "+FiLM",
    "L2": "+动作边界/层次聚合",
    "L3": "+嵌套实体边界",
    "L4": "+时间边",
    "L5": "+语义边",
    "L6": "+跨层消息",
    "L7": "+多正常原型",
    "L8": "+对比预训练",
    "L7b": "+原型多样/均衡",
}

ALL_OFF = {
    "use_hierarchy": False,
    "use_strategy_film": False,
    "use_action_boundary": False,
    "use_entity_boundary": False,
    "use_temporal_edges": False,
    "use_semantic_edges": False,
    "use_cross_level_messages": False,
    "use_prototypes": False,
}


def level_model_config(base: dict, level: str) -> dict:
    """Return the model config dict for a ladder level."""
    model = yaml.safe_load(
        Path(base).read_text(encoding="utf-8")
    )
    if "model" in model:
        model = model["model"]
    ablation = dict(ALL_OFF)
    ablation.update(LEVEL_ABLATIONS[level])
    model["ablation"] = ablation
    return model


def run_level(
    config_path: Path,
    model_config_path: Path,
    seed: int,
    tag: str,
    level: str,
    max_epochs: int,
    pretrain_epochs: int,
    mask_template_prob: float,
) -> int:
    command = [
        sys.executable,
        "scripts/train.py",
        "--config", str(config_path),
        "--model-config", str(model_config_path),
        "--seed", str(seed),
        "--tag", f"{tag}/{level}",
        "--max-epochs", str(max_epochs),
        "--pretrain-epochs", str(pretrain_epochs),
        "--mask-template-prob", str(mask_template_prob),
    ]
    if level != "L8" or pretrain_epochs <= 0:
        command.append("--skip-pretrain")
    if level == "L7b":
        command.append("--prototype-diversity")
    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"
    return subprocess.call(command, env=environment)


def load_result(output_dir: Path, tag: str, level: str, seed: int) -> dict | None:
    path = output_dir / f"{tag}/{level}" / f"seed_{seed}" / "result.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("test") or {}
    return {
        "seed": seed,
        "epochs": payload.get("epochs"),
        "val_auprc": payload.get("best_metric"),
        "test_auprc": test.get("auprc"),
        "test_auroc": test.get("auroc"),
        "test_f1": test.get("f1"),
        "pretrained": bool(payload.get("pretraining")),
    }


def fmt(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "N/A"
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f}±{arr.std(ddof=1):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiment config (per dataset)")
    parser.add_argument("--model-config", default="configs/model/sdhtg.yaml")
    parser.add_argument(
        "--levels",
        default="L0,L1,L2,L3,L4,L5,L6,L7,L8",
        help="comma-separated ladder levels",
    )
    parser.add_argument("--seeds", default="42,123", help="comma-separated seeds")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--mask-template-prob", type=float, default=0.0)
    parser.add_argument("--tag", default="ladder")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    output_root = Path(args.output_root)
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    scratch = output_root / "_ladder" / output_dir.name / args.tag
    scratch.mkdir(parents=True, exist_ok=True)

    summary = {}
    previous = None
    for level in levels:
        if level not in LEVEL_ABLATIONS:
            raise SystemExit(f"unknown level {level!r}")
        model_config = level_model_config(args.model_config, level)
        model_path = scratch / f"model_{level}.yaml"
        model_path.write_text(
            yaml.safe_dump({"model": model_config}, allow_unicode=True),
            encoding="utf-8",
        )
        rows = []
        for seed in seeds:
            result = load_result(output_dir, args.tag, level, seed)
            if result is not None and args.skip_existing:
                rows.append(result)
                print(f"  {level} seed {seed}: cached")
                continue
            print(f"  {level} seed {seed}: running ...", flush=True)
            code = run_level(
                config_path,
                model_path,
                seed,
                args.tag,
                level,
                args.max_epochs,
                args.pretrain_epochs if level == "L8" else 0,
                args.mask_template_prob,
            )
            if code != 0:
                print(f"  !! {level} seed {seed} failed with exit code {code}")
                continue
            result = load_result(output_dir, args.tag, level, seed)
            if result is not None:
                rows.append(result)

        if rows:
            summary[level] = rows
            delta = ""
            if previous is not None and previous:
                prev_mean = float(np.mean([r["test_auprc"] for r in previous]))
                cur_mean = float(np.mean([r["test_auprc"] for r in rows]))
                delta = f"  (ΔAUPRC {cur_mean - prev_mean:+.4f})"
            print(
                f"[{level}] {LEVEL_LABELS[level]}: "
                f"valAUPRC {fmt([r['val_auprc'] for r in rows])}  "
                f"testAUPRC {fmt([r['test_auprc'] for r in rows])}  "
                f"F1 {fmt([r['test_f1'] for r in rows])}{delta}"
            )
            previous = rows

    out_path = scratch / "ladder_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
