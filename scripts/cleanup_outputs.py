# -*- coding: utf-8 -*-
"""
cleanup_outputs.py -- remove redundant experiment artifacts.

Rule A  delete `last.pt` of COMPLETED runs (a result.json exists in the same
        seed directory). `best.pt`, metrics and thresholds are kept, so no
        reported result becomes unverifiable. `last.pt` is only needed to
        resume an interrupted training job.
Rule B  delete whole tag directories that are stale, invalid or superseded:
        July legacy single-run layout, pre-protocol structure ablations,
        superseded / mis-configured ladders, debug smoke runs, the redundant
        reference variant ablation_shortcuts_full, and the superseded
        pretraining-augmentation trials (*_aux / *_aux2).

Safety: directories modified within --min-age-minutes are skipped (a running
job may still be writing), and the script is dry-run unless --apply is given.

Usage:
    python scripts/cleanup_outputs.py            # dry run
    python scripts/cleanup_outputs.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


STALE_TAG_PATTERNS = (
    "ladder_smoke", "ladder_check", "ladder_v1", "ladder_iter", "ladder_proto",
    "ladder_dir1", "ladder_dir2", "ladder_dir3", "ladder_dir4",
    "ladder_ssh", "prof1", "prof2", "exitcheck", "lf_smoke",
    "l0_dbg", "l2_aux_dbg", "l2_aux_dbg2", "l2_aux_dbg3", "l2_aux_dbg4",
    "smoke_tcn", "smoke_transformer", "smoke_gnn_flat",
    "ablation_shortcuts_full",
    "ladder_l2_aux", "ladder_l7_aux", "ladder_l2_aux2", "ladder_l7_aux2",
)

STALE_STRUCTURE_GLOBS = ("*/struct_*", "*/main/struct_*")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def is_recent(path: Path, minutes: float) -> bool:
    if path.is_file():
        newest = path.stat().st_mtime
    else:
        try:
            newest = max(f.stat().st_mtime for f in path.rglob("*") if f.is_file())
        except ValueError:
            return False
    return (time.time() - newest) < minutes * 60


def collect_targets(root: Path) -> list[tuple[Path, int, str]]:
    planned: list[tuple[Path, int, str]] = []
    for last in root.rglob("last.pt"):
        # checkpoints live in <seed_dir>/checkpoints/last.pt
        seed_dir = last.parent.parent
        completed = (seed_dir / "result.json").is_file()
        pretraining_only = (last.parent / "pretrain_history.json").is_file()
        if completed or pretraining_only:
            planned.append((last, last.stat().st_size, "A:last.pt(completed)"))
    for pattern in STALE_STRUCTURE_GLOBS:
        for path in root.glob(pattern):
            if path.is_dir():
                planned.append((path, dir_size(path), f"B:struct({path.name})"))
    for path in root.glob("*/*/*"):
        if not path.is_dir():
            continue
        if path.name.startswith("seed_") or path.name in STALE_TAG_PATTERNS:
            planned.append((path, dir_size(path), f"B:{path.name}"))

    # drop children already covered by a planned ancestor directory
    targets = sorted({p for p, _, _ in planned}, key=lambda p: len(p.parts))
    final: list[tuple[Path, int, str]] = []
    for path, size, reason in planned:
        if any(path != other and other in path.parents for other in targets):
            continue
        final.append((path, size, reason))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-age-minutes", type=float, default=30.0)
    args = parser.parse_args()

    final = collect_targets(Path(args.root))
    freed = 0
    deleted = 0
    for path, size, reason in sorted(final, key=lambda x: -x[1]):
        if is_recent(path, args.min_age_minutes):
            print(f"SKIP (active) {path}")
            continue
        freed += size
        deleted += 1
        print(
            f"{'DELETE' if args.apply else 'WOULD DELETE'} "
            f"{size / 1024**2:9.1f} MB  {reason:<26} {path}"
        )
        if args.apply:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    print(
        f"\n{'freed' if args.apply else 'would free'}: {freed / 1024**3:.1f} GB "
        f"({deleted} targets)"
    )
    if not args.apply:
        print("re-run with --apply to delete")


if __name__ == "__main__":
    main()
