# -*- coding: utf-8 -*-
"""
wait_for_pretraining.py -- block until another pipeline step has finished.

``run_after_pretrain.bat`` uses this so the post-pretraining stages never start
a second GPU job while ``run_pretraining.py`` is still working. It polls the
process table for command lines matching the given patterns and returns as soon
as none is left.

Usage:
    python scripts/wait_for_pretraining.py
    python scripts/wait_for_pretraining.py --patterns run_pretraining.py --poll 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import psutil


SELF = os.path.basename(__file__)


def matches(patterns: list[str]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        joined = " ".join(process.info.get("cmdline") or [])
        if not joined or SELF in joined:
            continue
        if any(pattern in joined for pattern in patterns):
            hits.append((process.info["pid"], joined[:140]))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["run_pretraining.py"],
        help="substrings that identify the processes to wait for",
    )
    parser.add_argument("--poll", type=float, default=300.0,
                        help="seconds between checks")
    parser.add_argument("--label", default="upstream job",
                        help="name used in the progress messages")
    parser.add_argument("--timeout", type=float, default=None,
                        help="give up after this many minutes (default: never)")
    args = parser.parse_args()

    started = time.monotonic()
    while True:
        hits = matches(args.patterns)
        if not hits:
            print(f"no {args.label} process is running - continuing", flush=True)
            return
        waited = (time.monotonic() - started) / 60.0
        print(
            f"{args.label} still running: pid={hits[0][0]} "
            f"({len(hits)} match(es), waited {waited:.1f} min) - "
            f"re-checking in {args.poll / 60.0:.0f} min",
            flush=True,
        )
        if args.timeout is not None and waited >= args.timeout:
            print(f"timeout of {args.timeout:.0f} min reached - giving up")
            sys.exit(1)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
