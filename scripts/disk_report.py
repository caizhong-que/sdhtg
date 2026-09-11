# -*- coding: utf-8 -*-
"""Report disk usage of experiment outputs by dataset / tag."""
from __future__ import annotations

import argparse
from pathlib import Path


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        candidates = []
        main = dataset_dir / "main"
        if main.is_dir():
            candidates.append(("main", main))
        candidates += [
            (d.name, d)
            for d in sorted(dataset_dir.iterdir())
            if d.is_dir() and d.name != "main"
        ]
        for label, parent in candidates:
            for tag in sorted(parent.iterdir()):
                if not tag.is_dir():
                    continue
                size = dir_size(tag)
                runs = len(list(tag.rglob("result.json")))
                checkpoints = len(list(tag.rglob("*.pt")))
                rows.append(
                    (size / 1024**3, f"{dataset_dir.name}/{label}/{tag.name}", runs, checkpoints)
                )

    rows.sort(reverse=True)
    print(f"{'GB':>8}  {'tag':<56}{'runs':>6}{'ckpt':>6}")
    for size_gb, name, runs, checkpoints in rows[: args.top]:
        print(f"{size_gb:8.2f}  {name:<56}{runs:>6}{checkpoints:>6}")
    total = sum(r[0] for r in rows)
    print(f"\ntags={len(rows)}  total={total:.1f} GB")

    checkpoint_total = sum(
        f.stat().st_size
        for f in root.rglob("*.pt")
        if f.is_file()
    )
    last_total = sum(
        f.stat().st_size for f in root.rglob("last.pt") if f.is_file()
    )
    best_total = sum(
        f.stat().st_size for f in root.rglob("best.pt") if f.is_file()
    )
    print(
        f"checkpoints: all={checkpoint_total / 1024**3:.1f} GB  "
        f"best={best_total / 1024**3:.1f} GB  "
        f"last={last_total / 1024**3:.1f} GB"
    )


if __name__ == "__main__":
    main()
