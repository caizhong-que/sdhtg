from __future__ import annotations

import pandas as pd


def temporal_session_split(events: pd.DataFrame, ratios=(0.6, 0.2, 0.2), random_seed: int | None = None) -> pd.DataFrame:
    sessions = (events.groupby("session_id", sort=False)
                .agg(start=("timestamp", "min"), end=("timestamp", "max"))
                .sort_values(["start", "end"], kind="mergesort")
                if random_seed is None
                else events.groupby("session_id", sort=False)
                .agg(start=("timestamp", "min"), end=("timestamp", "max"))
                .sample(frac=1, random_state=random_seed))
    count = len(sessions)
    if count < 3:
        raise ValueError("at least three sessions are required for train/validation/test")
    first = max(1, int(count * ratios[0]))
    second = max(first + 1, int(count * (ratios[0] + ratios[1])))
    second = min(second, count - 1)
    mapping = {}
    for index, session_id in enumerate(sessions.index):
        mapping[session_id] = "train" if index < first else "validation" if index < second else "test"
    out = events.copy()
    out["split"] = out.session_id.map(mapping)
    if out.split.isna().any():
        raise RuntimeError("failed to assign split")
    assert_no_session_leakage(out)
    return out


def assert_no_session_leakage(events: pd.DataFrame) -> None:
    counts = events.groupby("session_id").split.nunique()
    if (counts != 1).any():
        raise ValueError("session leakage across temporal partitions")
