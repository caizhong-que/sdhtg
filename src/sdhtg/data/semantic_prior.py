from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ACTION = re.compile(
    r"\b(start|stop|open|close|read|write|create|delete|connect|disconnect|"
    r"authenticate|authorize|mount|unmount|replicate|allocate|release|send|receive|"
    r"load|store|update|recover|restart|submit|complete)\w*\b", re.I
)
STATUS = re.compile(
    r"\b(success|successful|succeed|failed|failure|error|timeout|denied|complete|"
    r"completed|warning|retry|ready|running|started|stopped|unavailable|invalid)\w*\b", re.I
)


def _extract(template: str, regex: re.Pattern[str]) -> str:
    match = regex.search(template)
    return match.group(0).lower() if match else "UNK"


def attach_semantics(events: pd.DataFrame, override_file: Path | None = None) -> pd.DataFrame:
    overrides = {}
    if override_file and override_file.exists():
        overrides = json.loads(override_file.read_text(encoding="utf-8"))
    out = events.copy()
    entities, actions, statuses = [], [], []
    for row in out.itertuples():
        override = overrides.get(str(row.drain_cluster_id), {})
        entities.append(str(override.get("entity", row.entity or "UNK")))
        actions.append(str(override.get("action", _extract(row.template, ACTION))))
        # Status preserves one-to-one template identity when no reviewed abstraction exists.
        statuses.append(str(override.get("status", _extract(row.template, STATUS)
                                         if _extract(row.template, STATUS) != "UNK"
                                         else f"template:{row.drain_cluster_id}")))
    out["entity_sem"] = entities
    out["action_sem"] = actions
    out["status_sem"] = statuses
    return out
