# -*- coding: utf-8 -*-
"""audit_cache_provenance.py -- build time and code version of each cache."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


DATASETS = ["bgl", "hdfs", "openstack", "ssh", "thunderbird", "bgl_temporal"]


def main():
    for dataset in DATASETS:
        path = Path(f"data/processed/{dataset}/manifest.json")
        if not path.is_file():
            print("{:<14} no manifest".format(dataset))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        commit = payload["git_commit"]
        date = subprocess.run(["git", "show", "-s", "--format=%ci", commit],
                              capture_output=True, text=True).stdout.strip()
        sessions = [o["sha256"] for o in payload["outputs"]
                    if o["path"].endswith("sessions.parquet")]
        print("{:<14} built={}  commit={} ({})  sessions={}".format(
            dataset, payload["created_utc"][:19], commit[:10], date,
            sessions[0][:12] if sessions else "?"))

    print("")
    for commit, label in (("5156c6f", "split_seed added"),
                          ("8acdf59", "data protocol refactor"),
                          ("a35db7ff", "SSH cache build commit")):
        date = subprocess.run(["git", "show", "-s", "--format=%ci %s", commit],
                              capture_output=True, text=True).stdout.strip()
        print("{:<10} {} -> {}".format(commit, label, date))


if __name__ == "__main__":
    main()
