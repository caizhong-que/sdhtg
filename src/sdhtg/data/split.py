from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _group_sessions_by_shared_events(
    events: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Union sessions that share a source event (HDFS block replication).

    A single raw event mentioning several block IDs is duplicated into several
    sessions. Such sessions must never be split across partitions, otherwise
    the same raw event leaks into train and test. Returns a session->group map
    and a per-group frame with start/end times.
    """
    if "source_event_id" not in events.columns:
        raise ValueError("group_by_source_event requires source_event_id")
    event_sessions: dict[int, set[str]] = defaultdict(set)
    for session_id, source_id in events[["session_id", "source_event_id"]].itertuples(index=False):
        event_sessions[source_id].add(session_id)

    parent: dict[str, str] = {}

    def find(session: str) -> str:
        while parent.get(session, session) != session:
            parent[session] = parent.get(parent[session], parent[session])
            session = parent[session]
        return session

    def union(left: str, right: str) -> None:
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    for sessions in event_sessions.values():
        sessions = list(sessions)
        for index in range(1, len(sessions)):
            union(sessions[0], sessions[index])

    session_to_group = {
        session: find(session) for session in events["session_id"].unique()
    }
    grouped = events.copy()
    grouped["_split_group"] = grouped.session_id.map(session_to_group)
    group_stats = (
        grouped.groupby("_split_group", sort=False)
        .agg(start=("timestamp", "min"), end=("timestamp", "max"))
        .sort_values(["start", "end"], kind="mergesort")
    )
    return session_to_group, group_stats


def temporal_session_split(
    events: pd.DataFrame,
    ratios=(0.6, 0.2, 0.2),
    random_seed: int | None = None,
    group_by_source_event: bool = False,
) -> pd.DataFrame:
    session_to_group = None
    grouped_sessions = None
    if group_by_source_event:
        session_to_group, grouped_sessions = _group_sessions_by_shared_events(events)
        units = grouped_sessions
    else:
        sessions = events.groupby("session_id", sort=False).agg(
            start=("timestamp", "min"), end=("timestamp", "max")
        )
        units = (
            sessions.sort_values(["start", "end"], kind="mergesort")
            if random_seed is None
            else sessions.sample(frac=1, random_state=random_seed)
        )

    count = len(units)
    if count < 3:
        raise ValueError("at least three sessions are required for train/validation/test")
    first = max(1, int(count * ratios[0]))
    second = max(first + 1, int(count * (ratios[0] + ratios[1])))
    second = min(second, count - 1)

    if group_by_source_event:
        group_to_split = {}
        for index, group_id in enumerate(units.index):
            group_to_split[group_id] = (
                "train" if index < first else "validation" if index < second else "test"
            )
        mapping = {
            session: group_to_split[group]
            for session, group in session_to_group.items()
        }
    else:
        mapping = {}
        for index, session_id in enumerate(units.index):
            mapping[session_id] = (
                "train" if index < first else "validation" if index < second else "test"
            )

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
