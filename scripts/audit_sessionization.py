# -*- coding: utf-8 -*-
"""Audit the regenerated sessionization for known risk patterns."""
import json
from pathlib import Path

import pandas as pd

from sdhtg.data.config import load_config
from sdhtg.data.sessionize import estimate_adaptive_idle_thresholds


def main() -> None:
    for dataset in ("bgl", "thunderbird"):
        config = load_config(f"configs/data/{dataset}.yaml")
        report_path = Path(f"data/processed/{dataset}/quality_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))

        events = pd.read_parquet(
            f"data/processed/{dataset}/events.parquet",
            columns=["entity", "split", "timestamp", "source_event_id"],
        )
        thresholds = estimate_adaptive_idle_thresholds(events, config)
        per_entity = thresholds["per_entity"]
        global_threshold = thresholds["global"]
        entity_values = events.entity.astype(str)
        covered = entity_values.isin(set(per_entity.keys()))
        print(f"[{dataset}] entities={events.entity.nunique()} "
              f"covered_in_dict={int(covered.sum())} "
              f"events_total={len(events)}")
        print(f"  global fallback threshold = {global_threshold}s")
        for split in ("train", "validation", "test"):
            mask = events.split == split
            print(f"  {split}: fallback-threshold events = "
                  f"{int((~covered & mask).sum())} / {int(mask.sum())} "
                  f"({(~covered & mask).mean():.4f})")

        sessions = pd.read_parquet(f"data/processed/{dataset}/sessions.parquet")
        print(f"  sessions={len(sessions)} truncated={int(sessions.truncated.sum())} "
              f"multi_chunk={int((sessions.chunk_index > 0).sum())}")


if __name__ == "__main__":
    main()
