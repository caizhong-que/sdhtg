# -*- coding: utf-8 -*-
"""Build every ablation/baseline model config and run a CPU forward pass."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ablations import GROUPS  # noqa: E402
from sdhtg.models.factory import build_model  # noqa: E402


def dummy_batch(batch_size: int = 2, steps: int = 8) -> dict:
    mask = torch.ones(batch_size, steps, dtype=torch.bool)
    return {
        "template_id": torch.randint(2, 50, (batch_size, steps)),
        "entity_id": torch.randint(2, 20, (batch_size, steps)),
        "action_id": torch.randint(2, 8, (batch_size, steps)),
        "status_id": torch.randint(2, 30, (batch_size, steps)),
        "delta_t": torch.rand(batch_size, steps),
        "action_change": (torch.rand(batch_size, steps) > 0.5).float(),
        "entity_change": (torch.rand(batch_size, steps) > 0.5).float(),
        "mask": mask,
        "label": torch.tensor([0.0, 1.0]),
    }


def main() -> None:
    overrides = {
        "template_vocab_size": 64,
        "entity_vocab_size": 32,
        "action_vocab_size": 16,
        "status_vocab_size": 64,
    }
    models = ["configs/model/tcn.yaml", "configs/model/transformer.yaml",
              "configs/model/gnn_flat.yaml"]
    for group, variants in GROUPS.items():
        models.extend(path for _, path, _ in variants)
    seen = []
    for path in models:
        if path in seen:
            continue
        seen.append(path)
        try:
            model = build_model(path, overrides)
            output = model(dummy_batch())
            finite = torch.isfinite(output.anomaly_logit).all().item()
            print(f"OK   {path}  logit_finite={finite}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {path}  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
