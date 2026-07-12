from __future__ import annotations

import configparser
import pickle
import re
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


TOKEN_MASKS = [
    ("IP", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
    ("HEX", re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b")),
    ("NUM", re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?(?![\w])")),
]


def normalize_content(text: str) -> str:
    result = text
    for name, regex in TOKEN_MASKS:
        result = regex.sub(f"<{name}>", result)
    return " ".join(result.split())


def make_miner(depth: int, similarity_threshold: float, max_children: int) -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.drain_depth = depth
    cfg.drain_sim_th = similarity_threshold
    cfg.drain_max_children = max_children
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def fit_transform_drain(events: pd.DataFrame, output_state: Path, settings: dict) -> pd.DataFrame:
    miner = make_miner(int(settings.get("depth", 4)),
                       float(settings.get("similarity_threshold", 0.5)),
                       int(settings.get("max_children", 100)))
    out = events.copy()
    out["normalized_content"] = out.content.map(normalize_content)
    train = out[out.split == "train"].sort_values(["timestamp", "source_event_id"])
    if train.empty:
        raise ValueError("empty training split")
    for message in train.normalized_content:
        miner.add_log_message(message)
    # Freeze: match() does not add clusters for validation/test.
    cluster_ids = []
    templates = []
    unseen = []
    for message in out.normalized_content:
        cluster = miner.match(message)
        if cluster is None:
            cluster_ids.append(0)
            templates.append("<UNK>")
            unseen.append(True)
        else:
            cluster_ids.append(int(cluster.cluster_id))
            templates.append(cluster.get_template())
            unseen.append(False)
    out["drain_cluster_id"] = cluster_ids
    out["template"] = templates
    out["unseen_template"] = unseen
    state = {
        "settings": settings,
        "clusters": [(int(c.cluster_id), c.get_template(), int(c.size))
                     for c in miner.drain.clusters],
    }
    output_state.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    return out
