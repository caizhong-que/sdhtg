# -*- coding: utf-8 -*-
"""
make_synthetic_structural.py -- structure-only anomaly dataset.

Every event comes from normal BGL sessions, so content shortcuts cannot solve
the task. Anomalies are created by inserting a foreign block of events from a
second normal session into the middle of a first session (workflow swap). The
true junction positions are known and exported for Boundary F1 evaluation.

Output (data/processed/bgl_synth):
    sessions.parquet  -- same schema as the standard cache
    vocab.json        -- rebuilt from the synthetic training split only
    boundaries.json   -- sample_id -> [junction event indices] (anomalies)

Usage:
    python scripts/make_synthetic_structural.py --out data/processed/bgl_synth
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sdhtg.data.cache import build_vocab, encode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/bgl_synth")
    parser.add_argument("--n-sessions", type=int, default=4000)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-sessions", default="data/processed/bgl_iter/sessions.parquet"
    )
    parser.add_argument(
        "--source-events", default="data/processed/bgl/events.parquet"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    sessions = pd.read_parquet(args.source_sessions)
    normal = sessions[
        (sessions.split == "train")
        & (sessions.label == 0)
        & (sessions.length >= args.min_len)
        & (sessions.length <= args.max_len)
    ]
    if len(normal) < args.n_sessions:
        raise SystemExit(
            f"only {len(normal)} normal sessions in range; "
            f"reduce --n-sessions"
        )
    selected = normal.sample(
        n=args.n_sessions, random_state=args.seed
    ).reset_index(drop=True)

    events = pd.read_parquet(
        args.source_events,
        columns=[
            "source_event_id",
            "entity_sem",
            "action_sem",
            "status_sem",
            "template",
            "timestamp",
        ],
    )
    wanted = set(
        int(x) for row in selected.source_event_ids for x in row
    )
    event_map = events[events.source_event_id.isin(wanted)].set_index(
        "source_event_id"
    )
    # Normal per-session event list, keyed by the session's dominant entity.
    session_events: list[tuple[str, list[pd.Series]]] = []
    for row in selected.itertuples(index=False):
        ids = [int(x) for x in row.source_event_ids]
        rows = event_map.loc[ids]
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        rows = rows.reset_index()  # restore source_event_id as a column
        entity = rows.entity_sem.mode().iloc[0]
        session_events.append((str(entity), [r for _, r in rows.iterrows()]))

    # Pooled intra-session delta distribution (used to hide junctions).
    normal_deltas = np.diff(
        np.sort(
            event_map.timestamp.astype("int64").to_numpy() / 1e9
        )
    )
    normal_deltas = normal_deltas[normal_deltas > 0]

    by_entity: dict[str, list[tuple[str, list[pd.Series]]]] = {}
    for entity, rows in session_events:
        by_entity.setdefault(entity, []).append((entity, rows))

    def build_sequence(
        entries: list[tuple[str, list[pd.Series]]],
    ) -> tuple[list[pd.Series], list[int]]:
        """Normal: original session. Anomalous: insert B into A."""
        a_entity, a_rows = entries[0]
        same = [e for e in entries[1:] if e[0] == a_entity]
        pool = same if same else entries[1:]
        candidates = rng.sample(pool, min(len(pool), 10))
        # Prefer junctions that align with action changes so the semantic
        # change prior/auxiliary supervision can localize the swap.
        for _, b_rows in candidates:
            for split in range(1, len(a_rows)):
                if (
                    str(a_rows[split - 1].action_sem)
                    != str(b_rows[0].action_sem)
                    and str(b_rows[-1].action_sem)
                    != str(a_rows[split].action_sem)
                ):
                    sequence = a_rows[:split] + b_rows + a_rows[split:]
                    return sequence, [split, split + len(b_rows)]
        # Fallback: mid-split insertion (junctions may not align with actions).
        _, b_rows = candidates[0]
        split = max(1, len(a_rows) // 2)
        sequence = a_rows[:split] + b_rows + a_rows[split:]
        return sequence, [split, split + len(b_rows)]

    records = []
    boundaries: dict[str, list[int]] = {}
    sample_index = 0

    def add_sample(sequence: list[pd.Series], label: int, junctions: list[int]):
        nonlocal sample_index
        sample_id = f"synth:{sample_index}"
        sample_index += 1
        delta = [
            float(np_rng.choice(normal_deltas))
            if i > 0
            else 0.0
            for i in range(len(sequence))
        ]
        # Junction deltas are hidden in the pooled normal distribution.
        for j in junctions:
            delta[j] = float(np_rng.choice(normal_deltas))
        action = [str(r.action_sem) for r in sequence]
        entity = [str(r.entity_sem) for r in sequence]
        records.append(
            {
                "sample_id": sample_id,
                "session_id": f"synth_sess:{sample_index}",
                "chunk_index": 0,
                "split": "",
                "label": label,
                "start_timestamp": 0,
                "end_timestamp": float(np.sum(delta)),
                "source_event_ids": [int(r.source_event_id) for r in sequence],
                "template": [str(r.template) for r in sequence],
                "entity_sem": entity,
                "action_sem": action,
                "status_sem": [str(r.status_sem) for r in sequence],
                "delta_t": delta,
                "action_change": [0.0]
                + [float(a != b) for a, b in zip(action[:-1], action[1:])],
                "entity_change": [0.0]
                + [float(a != b) for a, b in zip(entity[:-1], entity[1:])],
                "length": len(sequence),
                "truncated": False,
            }
        )
        if label == 1:
            boundaries[sample_id] = junctions

    shuffled = list(session_events)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    for index, entry in enumerate(shuffled[:half]):
        add_sample(entry[1], 0, [])
    for index in range(half):
        a_entry = shuffled[index]
        # Pair with another entry, avoiding identity.
        partner = shuffled[(index + half) % len(shuffled)]
        if partner is a_entry:
            partner = shuffled[(index + 1) % len(shuffled)]
        sequence, junctions = build_sequence([a_entry, partner])
        add_sample(sequence, 1, junctions)

    frame = pd.DataFrame(records)
    rng.shuffle(records)
    # Deterministic 60/20/20 split over shuffled samples.
    n = len(records)
    first = int(n * 0.6)
    second = int(n * 0.8)
    for i, record in enumerate(records):
        record["split"] = (
            "train" if i < first else "validation" if i < second else "test"
        )
    frame = pd.DataFrame(records)

    # Vocabulary from the synthetic training split only.
    train_events = frame[frame.split == "train"]
    def flatten(series: pd.Series) -> pd.Series:
        return pd.Series([x for values in series for x in values])

    vocabs = {
        "template": build_vocab(flatten(train_events.template)),
        "entity": build_vocab(flatten(train_events.entity_sem)),
        "action": build_vocab(flatten(train_events.action_sem)),
        "status": build_vocab(flatten(train_events.status_sem)),
    }

    out_frame = pd.DataFrame(
        {
            "sample_id": frame.sample_id,
            "session_id": frame.session_id,
            "chunk_index": frame.chunk_index,
            "split": frame.split,
            "label": frame.label,
            "start_timestamp": frame.start_timestamp,
            "end_timestamp": frame.end_timestamp,
            "source_event_ids": frame.source_event_ids,
            "template_ids": [
                [encode(x, vocabs["template"]) for x in v] for v in frame.template
            ],
            "entity_ids": [
                [encode(x, vocabs["entity"]) for x in v] for v in frame.entity_sem
            ],
            "action_ids": [
                [encode(x, vocabs["action"]) for x in v] for v in frame.action_sem
            ],
            "status_ids": [
                [encode(x, vocabs["status"]) for x in v] for v in frame.status_sem
            ],
            "delta_t": frame.delta_t,
            "action_change": frame.action_change,
            "entity_change": frame.entity_change,
            "length": frame.length,
            "truncated": frame.truncated,
        }
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_frame.to_parquet(out_dir / "sessions.parquet", index=False, compression="zstd")
    (out_dir / "vocab.json").write_text(
        json.dumps(vocabs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "boundaries.json").write_text(
        json.dumps(boundaries, indent=2), encoding="utf-8"
    )
    stats = {
        split: {
            "samples": int((out_frame.split == split).sum()),
            "anomalous": int(
                ((out_frame.split == split) & (out_frame.label == 1)).sum()
            ),
        }
        for split in ("train", "validation", "test")
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
