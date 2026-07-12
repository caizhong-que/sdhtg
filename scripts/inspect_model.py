from __future__ import annotations

import argparse
import json

import torch

from sdhtg.models.factory import build_model


def synthetic_batch(batch_size: int, steps: int, device: str) -> dict[str, torch.Tensor]:
    mask = torch.ones(batch_size, steps, dtype=torch.bool, device=device)
    return {
        "template_id": torch.randint(2, 128, (batch_size, steps), device=device),
        "entity_id": torch.randint(2, 32, (batch_size, steps), device=device),
        "action_id": torch.randint(2, 32, (batch_size, steps), device=device),
        "status_id": torch.randint(2, 128, (batch_size, steps), device=device),
        "delta_t": torch.rand(batch_size, steps, device=device) * 30.0,
        "action_change": torch.randint(0, 2, (batch_size, steps), device=device).float(),
        "entity_change": torch.randint(0, 2, (batch_size, steps), device=device).float(),
        "mask": mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/sdhtg.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    model = build_model(args.config).to(args.device).eval()
    batch = synthetic_batch(args.batch_size, args.steps, args.device)
    with torch.no_grad():
        output = model(batch)
    print(json.dumps(model.parameter_report(), indent=2))
    print("anomaly_probability:", output.anomaly_probability.cpu().tolist())
    print("graph metadata:", output.graph_batch.metadata())


if __name__ == "__main__":
    main()
