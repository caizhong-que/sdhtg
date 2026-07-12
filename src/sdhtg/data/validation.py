from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_EVENT_COLUMNS = {
    "source_event_id", "timestamp", "content", "entity", "event_label",
    "session_id", "session_label", "split", "template", "drain_cluster_id",
    "entity_sem", "action_sem", "status_sem",
}


def validate_processed(events: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"event cache missing columns: {sorted(missing)}")
    if events.empty or sessions.empty:
        raise ValueError("processed events/sessions must not be empty")
    if events[list(REQUIRED_EVENT_COLUMNS)].isna().any().any():
        raise ValueError("required processed event columns contain nulls")
    if not set(events.event_label.unique()).issubset({0, 1}):
        raise ValueError("non-binary event labels")
    if not set(events.session_label.unique()).issubset({0, 1}):
        raise ValueError("non-binary session labels")
    leakage = events.groupby("session_id").split.nunique()
    if (leakage != 1).any():
        raise ValueError("session leakage detected")
    if (sessions.length <= 0).any():
        raise ValueError("empty session chunks detected")
    if sessions.sample_id.duplicated().any():
        raise ValueError("duplicate sample_id")
    split_stats = {}
    for split in ("train", "validation", "test"):
        e = events[events.split == split]
        s = sessions[sessions.split == split]
        if e.empty or s.empty:
            raise ValueError(f"empty required split: {split}")
        split_stats[split] = {
            "events": int(len(e)), "sessions": int(len(s)),
            "anomalous_events": int(e.event_label.sum()),
            "anomalous_sessions": int(s.label.sum()),
            "unseen_template_rate": float(e.unseen_template.mean()),
        }
    return {
        "events": int(len(events)), "sessions": int(len(sessions)),
        "templates": int(events.drain_cluster_id.nunique()),
        "entities": int(events.entity_sem.nunique()),
        "actions": int(events.action_sem.nunique()),
        "statuses": int(events.status_sem.nunique()),
        "duplicate_source_memberships": int(events.source_event_id.duplicated().sum()),
        "splits": split_stats,
    }


def write_quality_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
