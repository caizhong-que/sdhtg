"""Rebuild out-of-bounds session IDs after a vocab drift.

The pre-vocab-fix ``build_vocab`` could let a real token equal to ``<PAD>`` or
``<UNK>`` overwrite the sentinel ID, producing encoded IDs that are >= vocab
size.  This script re-encodes the affected field(s) from ``events.parquet`` and
rewrites ``sessions.parquet`` and ``vocab.json`` for one processed dataset.

Usage:
    python scripts/repair_processed_vocab.py --dataset thunderbird
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sdhtg.data.cache import build_vocab, encode


FIELD_MAP = {
    "template": ("template", "template_ids"),
    "entity": ("entity_sem", "entity_ids"),
    "action": ("action_sem", "action_ids"),
    "status": ("status_sem", "status_ids"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed-dir", default=None)
    args = parser.parse_args()

    processed = (
        Path(args.processed_dir)
        if args.processed_dir
        else Path("data/processed") / args.dataset
    )
    events = pd.read_parquet(processed / "events.parquet")
    sessions = pd.read_parquet(processed / "sessions.parquet")
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))

    train = events[events.split == "train"]
    repaired = []

    for vocab_name, (event_field, session_field) in FIELD_MAP.items():
        current_vocab = vocab[vocab_name]
        size = len(current_vocab)
        max_id = max(max(ids) for ids in sessions[session_field])
        if max_id < size:
            continue

        new_vocab = build_vocab(train[event_field])
        event_id_to_value = dict(
            zip(events["source_event_id"], events[event_field].astype(str))
        )
        sessions[session_field] = sessions["source_event_ids"].apply(
            lambda ids: [
                encode(event_id_to_value[int(event_id)], new_vocab)
                for event_id in ids
            ]
        )
        vocab[vocab_name] = new_vocab
        repaired.append(f"{vocab_name}: {size} -> {len(new_vocab)}")

    if not repaired:
        print("No out-of-bounds field found; nothing changed.")
        return

    sessions.to_parquet(processed / "sessions.parquet", index=False, compression="zstd")
    (processed / "vocab.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Repaired:", "; ".join(repaired))


if __name__ == "__main__":
    main()
