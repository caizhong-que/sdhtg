from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from .config import SDHTGModelConfig
from .baselines import (FlatGraphBaseline, MaskedTemplateTransformer,
                        TCNBaseline, TransformerBaseline)
from .sdhtg import SDHTG


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = mapping
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"override path crosses a scalar at {key!r}")
        cursor = child
    cursor[keys[-1]] = value


def load_model_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> SDHTGModelConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model configuration must be a YAML mapping")
    raw = deepcopy(raw)
    for key, value in (overrides or {}).items():
        normalized = key.removeprefix("model.")
        _set_nested(raw.setdefault("model", {}), normalized, value)
    return SDHTGModelConfig.from_dict(raw)


def build_model(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    config = load_model_config(path, overrides)
    if config.arch == "tcn":
        return TCNBaseline(config)
    if config.arch == "transformer":
        return TransformerBaseline(config)
    if config.arch == "gnn_flat":
        return FlatGraphBaseline(config)
    if config.arch == "masked_template":
        return MaskedTemplateTransformer(config)
    return SDHTG(config)
