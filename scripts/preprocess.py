from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdhtg.data.adapters import create_adapter
from sdhtg.data.cache import build_vocabs, materialize_sessions, write_cache
from sdhtg.data.config import load_config
from sdhtg.data.drain_parser import fit_transform_drain
from sdhtg.data.integrity import validate_files, write_manifest
from sdhtg.data.semantic_prior import attach_semantics
from sdhtg.data.sessionize import assign_sessions
from sdhtg.data.split import temporal_session_split
from sdhtg.data.validation import validate_processed, write_quality_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-missing-hash", action="store_true")
    parser.add_argument("--semantic-overrides")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_records = validate_files(cfg, args.allow_missing_hash)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    events = create_adapter(cfg).normalize()
    events = assign_sessions(events, cfg)
    events = temporal_session_split(events, cfg.split, cfg.split_seed)
    drain_state = cfg.processed_dir / "drain_state.bin"
    events = fit_transform_drain(events, drain_state, cfg.drain)
    override = Path(args.semantic_overrides) if args.semantic_overrides else None
    events = attach_semantics(events, override)

    vocabs = build_vocabs(events)
    sessions = materialize_sessions(events, vocabs, cfg.max_session_length)
    outputs = write_cache(events, sessions, vocabs, cfg.processed_dir)
    outputs.append(drain_state)

    report = validate_processed(events, sessions)
    report_path = cfg.processed_dir / "quality_report.json"
    write_quality_report(report, report_path)
    outputs.append(report_path)

    manifest_path = cfg.processed_dir / "manifest.json"
    write_manifest(manifest_path, cfg, source_records, outputs)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
