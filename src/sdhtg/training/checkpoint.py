from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory: str | Path):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
    def save(self, name: str, *, model, optimizer, scheduler, scaler, epoch: int,
             global_step: int, best_metric: float, patience_count: int, metadata: dict) -> Path:
        path=self.directory/f"{name}.pt"; temporary=path.with_suffix(".pt.tmp")
        payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                 "scheduler":scheduler.state_dict() if scheduler else None,
                 "scaler":scaler.state_dict() if scaler else None,"epoch":epoch,"global_step":global_step,
                 "best_metric":best_metric,"patience_count":patience_count,"rng":rng_state(),"metadata":metadata}
        torch.save(payload,temporary); os.replace(temporary,path); return path
    def load(self, path, *, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict:
        payload=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(payload["model"])
        if optimizer is not None: optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None: scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload["scaler"] is not None: scaler.load_state_dict(payload["scaler"])
        restore_rng(payload["rng"]); return payload
