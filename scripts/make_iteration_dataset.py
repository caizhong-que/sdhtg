"""
make_iteration_dataset.py -- build a stratified subsample for fast iteration.

The full caches (BGL/Thunderbird) are too large for quick module-direction
checks. This script subsamples session chunks per split (preserving the
anomaly ratio), keeps only the events belonging to those chunks, rebuilds the
vocabulary from the subsampled *training* split only (same isolation protocol
as the full pipeline), and writes a standalone processed directory.

Usage:
    python scripts/make_iteration_dataset.py \
        --dataset bgl --sizes 25000,6000,6000 --seed 42 \
        --out data/processed/bgl_iter

The resulting directory can be used with a copy of the experiment config that
points ``data.processed_dir`` at it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sdhtg.data.cache import (
    PAD,
    UNK,
    build_vocab,
    materialize_sessions,
)


def subsample(frame: pd.DataFrame, size: int, seed: int, split: str) -> pd.DataFrame:
    part = frame[frame.split == split]
    if len(part) <= size:
        return part
    positive = part[part.label == 1]
    negative = part[part.label == 0]
    target_positive = max(1, int(round(size * len(positive) / len(part))))
    target_negative = size - target_positive
    rng = np.random.default_rng(seed)
    pos = positive.sample(
        n=min(target_positive, len(positive)), random_state=int(rng.integers(0, 2**31))
    )
    neg = negative.sample(
        n=min(target_negative, len(negative)), random_state=int(rng.integers(0, 2**31))
    )
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sizes", default="25000,6000,6000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    processed = Path(f"data/processed/{args.dataset}")
    sessions = pd.read_parquet(processed / "sessions.parquet")
    events = pd.read_parquet(processed / "events.parquet")
    sizes = [int(x) for x in args.sizes.split(",")]
    assert len(sizes) == 3, "sizes must be train,validation,test"

    pieces = [
        subsample(sessions, size, args.seed + offset, split)
        for size, split, offset in zip(sizes, ("train", "validation", "test"), (0, 1, 2))
    ]
    selected = pd.concat(pieces, ignore_index=True)
    selected_ids = set(
        int(x)
        for row in selected.source_event_ids
        for x in row
    )
    selected_events = events[events.source_event_id.isin(selected_ids)].copy()
    if selected_events.empty:
        raise SystemExit("no events selected; check source_event_ids coverage")

    # Rebuild vocabularies from the subsampled training split only.
    train_events = selected_events[selected_events.split == "train"]
    if train_events.empty:
        raise SystemExit("empty training split after subsampling")
    vocabs = {
        "template": build_vocab(train_events.template),
        "entity": build_vocab(train_events.entity_sem),
        "action": build_vocab(train_events.action_sem),
        "status": build_vocab(train_events.status_sem),
    }

    # Re-materialize the subsampled sessions with the NEW vocabulary so the
    # iteration cache follows the same isolation protocol as the full cache.
    out = materialize_sessions(selected_events, vocabs, 512)
    if out.empty:
        raise SystemExit("materialization produced no rows")
    out = out[out.split.isin(("train", "validation", "test"))].reset_index(drop=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "sessions.parquet", index=False, compression="zstd")
    (out_dir / "vocab.json").write_text(
        json.dumps(vocabs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stats = {
        split: {
            "samples": int((out.split == split).sum()),
            "anomalous": int(((out.split == split) & (out.label == 1)).sum()),
        }
        for split in ("train", "validation", "test")
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
