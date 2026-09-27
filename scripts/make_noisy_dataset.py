# -*- coding: utf-8 -*-
"""
make_noisy_dataset.py -- template-level parsing-noise variant of a dataset.

Corrupts the template ids of a processed cache to emulate parser errors, and
writes a derived cache that ``train.py`` can consume unchanged. The affected
fraction ``p`` is the probability that an individual template occurrence is
rewritten, so the four noise kinds are directly comparable:

    replace  rewrite to a uniformly random other template id
    merge    rewrite to the head of a frequency-ranked cluster (coarser templates)
    split    rewrite to an added sub-token (finer templates; the vocab grows)
    unk      rewrite to UNK

Protocols select which splits are corrupted:

    test_only  only the test split (drift that appears after training)
    all        train, validation and test alike

Usage:
    python scripts/make_noisy_dataset.py --source data/processed/ssh \
        --kind replace --protocol test_only --rate 0.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


KINDS = ("replace", "merge", "split", "unk")
PROTOCOLS = ("test_only", "all")
UNK = 1
PAD = 0


def _frequency_order(frame: pd.DataFrame) -> list[int]:
    counter: Counter[int] = Counter()
    for values in frame["template_ids"]:
        counter.update(int(v) for v in values)
    for special in (PAD, UNK):
        counter.pop(special, None)
    return [key for key, _ in counter.most_common()]


def _cluster_map(order: list[int], groups: int) -> dict[int, int]:
    """Frequency-ranked clustering: each template maps to its cluster head."""
    if not order:
        return {}
    groups = max(2, min(groups, len(order)))
    edges = np.linspace(0, len(order), groups + 1).astype(int)
    mapping: dict[int, int] = {}
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        head = order[start]
        for template in order[start:end]:
            mapping[template] = head
    return mapping


def _hash_seed(source: Path, kind: str, protocol: str, rate: float) -> int:
    payload = f"{source.resolve()}|{kind}|{protocol}|{rate:.4f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="processed dataset dir containing sessions.parquet")
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--target", default=None,
                        help="output dir (default: <source>_noise_<kind>_<protocol>)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target) if args.target else source.with_name(
        f"{source.name}_noise_{args.kind}_{args.protocol}"
    )
    if target.exists():
        if not args.overwrite:
            print(f"== {target} already exists (use --overwrite to rebuild)")
            return
        shutil.rmtree(target)
    target.mkdir(parents=True)

    frame = pd.read_parquet(source / "sessions.parquet")
    vocab = json.loads((source / "vocab.json").read_text(encoding="utf-8"))
    template_vocab = dict(vocab["template"])
    size = len(template_vocab)
    ids = sorted(template_vocab.values())
    rng = np.random.default_rng(_hash_seed(source, args.kind, args.protocol, args.rate))

    order = _frequency_order(frame)
    # Over-merging: rewrite to one of the most frequent templates, i.e. the
    # parser collapses rare templates onto coarse ones.
    head_count = max(2, int(round((1.0 - args.rate) * len(order))))
    heads = order[:head_count] or order
    if args.kind == "split":
        # Add two sub-tokens per original template; the model reads
        # template_vocab_size from vocab.json, so no code change is needed.
        for token, index in list(template_vocab.items()):
            template_vocab[f"<SPLIT1:{token}>"] = index + size
            template_vocab[f"<SPLIT2:{token}>"] = index + 2 * size
        vocab["template"] = template_vocab

    splits = ("test",) if args.protocol == "test_only" else ("train", "validation", "test")
    mask = frame["split"].isin(splits).to_numpy()
    corrupted: list[list[int]] = []
    affected = 0
    total = 0
    total_corrupted = 0
    for row, values in enumerate(frame["template_ids"]):
        items = [int(v) for v in values]
        total += len(items)
        if not mask[row]:
            corrupted.append(items)
            continue
        total_corrupted += len(items)
        take = rng.random(len(items)) < args.rate
        for position, hit in enumerate(take):
            if not hit or items[position] <= UNK:
                continue
            value = items[position]
            if args.kind == "replace":
                choices = [i for i in ids if i != value and i > UNK]
                items[position] = int(rng.choice(choices)) if choices else value
            elif args.kind == "merge":
                choices = [i for i in heads if i != value] or heads
                items[position] = int(rng.choice(choices))
            elif args.kind == "split":
                # Always re-emit as a different (finer) sub-token.
                items[position] = value + size * (1 + int(rng.random() < 0.5))
            else:
                items[position] = UNK
            affected += int(items[position] != value)
        corrupted.append(items)

    frame = frame.copy()
    frame["template_ids"] = corrupted
    frame.to_parquet(target / "sessions.parquet", index=False)
    (target / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    for extra in ("drain_state.bin",):
        candidate = source / extra
        if candidate.is_file():
            shutil.copy(candidate, target / extra)
    (target / "noise_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "target": str(target),
                "kind": args.kind,
                "protocol": args.protocol,
                "rate": args.rate,
                "corrupted_splits": list(splits),
                "template_occurrences": total,
                "template_occurrences_in_corrupted_splits": total_corrupted,
                "rewritten_occurrences": affected,
                "rewritten_fraction_corrupted_splits": affected / max(total_corrupted, 1),
                "rewritten_fraction_overall": affected / max(total, 1),
                "template_vocab_size": len(template_vocab),
                "seed": _hash_seed(source, args.kind, args.protocol, args.rate),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"== {target.name}: kind={args.kind} protocol={args.protocol} "
        f"rate={args.rate} rewritten={affected}/{total_corrupted} "
        f"({100 * affected / max(total_corrupted, 1):.2f}% of the corrupted split; "
        f"{100 * affected / max(total, 1):.2f}% of the dataset)"
    )


if __name__ == "__main__":
    main()
