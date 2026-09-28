# -*- coding: utf-8 -*-
"""make_entity_holdout.py -- entity-identity holdout cache for the grouping test.

The processed caches fit their vocabulary on the training split, so a test
entity that never appeared in training is stored as UNK.  This makes the natural
"unseen entity" group extremely small (0% on SSH/OpenStack, 0.28% on BGL, 0.30%
on Thunderbird) and undefined on HDFS (every test entity is UNK), i.e. the
natural grouping cannot support a quantitative comparison.

This script designs the grouping instead: it picks the entities that cover a
requested fraction of the *test* sessions and masks those identities to UNK in
every split (training, validation and test).  The model therefore never sees
those entities as identifiable, while the test sessions that carry them are
known from the manifest - so "identifiable entity" and "unidentifiable entity"
groups can be compared on the same weights and the same threshold.

Groups written to ``holdout_groups.csv`` (one row per session):

    holdout      session references at least one held-out entity
    natural_unk  no held-out entity, but at least one entity is UNK already
    seen         every referenced entity is identifiable

Usage:
    python scripts/make_entity_holdout.py --source data/processed/bgl \
        --coverage 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


PAD = 0
UNK = 1


def covering_entities(frame: pd.DataFrame, coverage: float) -> tuple[set[int], int]:
    test = frame[frame.split == "test"]
    counter: Counter[int] = Counter()
    for values in test["entity_ids"]:
        for entity in {int(v) for v in values}:
            if entity in (PAD, UNK):
                continue
            counter[entity] += 1
    target = coverage * len(test)
    selected: set[int] = set()
    covered = 0
    for entity, count in counter.most_common():
        if covered >= target:
            break
        selected.add(entity)
        covered += count
    return selected, covered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--coverage", type=float, default=0.2,
                        help="fraction of test sessions that should carry a held-out entity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    tag = f"entholdout{int(round(args.coverage * 100))}"
    target = Path(args.target) if args.target else source.with_name(f"{source.name}_{tag}")
    if target.exists():
        if not args.overwrite:
            print(f"== {target} already exists (use --overwrite to rebuild)")
            return
        shutil.rmtree(target)
    target.mkdir(parents=True)

    frame = pd.read_parquet(source / "sessions.parquet")
    held, covered = covering_entities(frame, args.coverage)
    print(f"== held-out entities: {len(held)} covering ~{covered} test sessions")

    groups, rewritten = [], []
    affected = 0
    for values in frame["entity_ids"]:
        tokens = [int(v) for v in values]
        current = {t for t in tokens if t not in (PAD, UNK)}
        is_holdout = bool(current & held)
        if is_holdout:
            groups.append("holdout")
        elif UNK in tokens:
            groups.append("natural_unk")
        else:
            groups.append("seen")
        masked = [
            UNK if (int(token) in held) else int(token) for token in tokens
        ]
        affected += sum(1 for a, b in zip(tokens, masked) if a != b)
        rewritten.append(masked)

    out = frame.copy()
    out["entity_ids"] = rewritten
    out.to_parquet(target / "sessions.parquet", index=False)
    shutil.copy(source / "vocab.json", target / "vocab.json")
    for extra in ("drain_state.bin",):
        candidate = source / extra
        if candidate.is_file():
            shutil.copy(candidate, target / extra)

    table = pd.DataFrame({
        "row": range(len(out)),
        "split": out["split"].to_numpy(),
        "group": groups,
        "label": out["label"].to_numpy(),
    })
    table.to_csv(target / "holdout_groups.csv", index=False)
    per_split = table.groupby(["split", "group"]).size().unstack(fill_value=0).to_dict()
    (target / "holdout_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "target": str(target),
                "coverage_requested": args.coverage,
                "tag": tag,
                "seed": args.seed,
                "held_out_entities": sorted(int(e) for e in held),
                "held_out_entity_count": len(held),
                "covered_test_sessions": covered,
                "masked_entity_tokens": affected,
                "per_split_group_counts": {k: {g: int(v) for g, v in vals.items()}
                                           for k, vals in per_split.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"== {target.name}: masked {affected} entity tokens; "
          f"groups per split = {json.dumps({k: v for k, v in per_split.items()}, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
