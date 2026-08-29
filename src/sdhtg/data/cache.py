from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(values: pd.Series) -> dict[str, int]:
    counts = Counter(map(str, values))
    ordered = sorted(counts, key=lambda x: (-counts[x], x))
    vocab = {PAD: 0, UNK: 1}
    next_id = 2
    for value in ordered:
        if value in (PAD, UNK):
            # Sentinel tokens are reserved; a real token with the same string
            # must not overwrite their IDs.
            continue
        vocab[value] = next_id
        next_id += 1
    return vocab


def encode(value: object, vocab: dict[str, int]) -> int:
    return vocab.get(str(value), vocab[UNK])


def build_vocabs(events: pd.DataFrame) -> dict[str, dict[str, int]]:
    train = events[events.split == "train"]
    if train.empty:
        raise ValueError("cannot build vocabulary from empty training split")
    return {
        "template": build_vocab(train.template),
        "entity": build_vocab(train.entity_sem),
        "action": build_vocab(train.action_sem),
        "status": build_vocab(train.status_sem),
    }


def chunk_session(group: pd.DataFrame, max_length: int) -> list[pd.DataFrame]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    return [group.iloc[start:start + max_length] for start in range(0, len(group), max_length)]


def materialize_sessions(events: pd.DataFrame, vocabs: dict, max_length: int) -> pd.DataFrame:
    records = []
    ordered = events.sort_values(["session_id", "timestamp", "source_event_id"], kind="mergesort")
    for session_id, group in ordered.groupby("session_id", sort=False):
        session_times = group.timestamp.astype("int64").to_numpy() / 1e9
        for chunk_index, chunk in enumerate(chunk_session(group, max_length)):
            start = chunk_index * max_length
            times = session_times[start:start + len(chunk)]
            # Preserve the inter-chunk gap: the first event of chunk > 0 keeps
            # its delta relative to the last event of the previous chunk.
            previous = (
                session_times[start - 1]
                if start > 0
                else times[0]
            )
            delta = np.diff(times, prepend=previous).clip(min=0)
            action = chunk.action_sem.astype(str).tolist()
            entity = chunk.entity_sem.astype(str).tolist()
            records.append({
                "sample_id": f"{session_id}:chunk:{chunk_index}",
                "session_id": str(session_id),
                "chunk_index": chunk_index,
                "split": str(chunk.split.iloc[0]),
                # Chunk label: max of the chunk's EVENT labels when event-level
                # labels exist (BGL/Thunderbird), identical to the session
                # label otherwise (HDFS/SSH per-key labels are session-uniform).
                "label": int(
                    chunk.event_label.max()
                    if "event_label" in chunk
                    else chunk.session_label.max()
                ),
                "start_timestamp": chunk.timestamp.iloc[0],
                "end_timestamp": chunk.timestamp.iloc[-1],
                "source_event_ids": chunk.source_event_id.astype(int).tolist(),
                "template_ids": [encode(x, vocabs["template"]) for x in chunk.template],
                "entity_ids": [encode(x, vocabs["entity"]) for x in entity],
                "action_ids": [encode(x, vocabs["action"]) for x in action],
                "status_ids": [encode(x, vocabs["status"]) for x in chunk.status_sem],
                "delta_t": delta.astype(float).tolist(),
                # First-event priors are 0 by definition (no previous event),
                # matching the manuscript's dA_1 = dE_1 = 0.
                "action_change": [0.0] + [float(a != b) for a, b in zip(action[:-1], action[1:])],
                "entity_change": [0.0] + [float(a != b) for a, b in zip(entity[:-1], entity[1:])],
                "length": len(chunk),
                "truncated": len(group) > max_length,
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("session materialization produced no samples")
    return frame


def write_cache(events: pd.DataFrame, sessions: pd.DataFrame, vocabs: dict, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    event_path = directory / "events.parquet"
    session_path = directory / "sessions.parquet"
    vocab_path = directory / "vocab.json"
    events.to_parquet(event_path, index=False, compression="zstd")
    sessions.to_parquet(session_path, index=False, compression="zstd")
    vocab_path.write_text(json.dumps(vocabs, indent=2, ensure_ascii=False), encoding="utf-8")
    return [event_path, session_path, vocab_path]
