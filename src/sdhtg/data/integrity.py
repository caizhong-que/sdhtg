from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import DataConfig, FileSpec


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_files(cfg: DataConfig, allow_missing_hash: bool = False) -> list[dict]:
    records = []
    for spec in cfg.files:
        path = cfg.raw_dir / spec.path
        if not path.is_file():
            raise FileNotFoundError(f"required {spec.role} file not found: {path}")
        actual = sha256_file(path)
        if not spec.sha256 and not allow_missing_hash:
            raise ValueError(f"SHA-256 missing for {path}; set it in YAML or explicitly allow it")
        if spec.sha256 and actual.lower() != spec.sha256.lower():
            raise ValueError(f"SHA-256 mismatch for {path}: expected {spec.sha256}, got {actual}")
        records.append({"path": str(path), "role": spec.role, "size": path.stat().st_size,
                        "sha256": actual, "hash_declared": bool(spec.sha256)})
    return records


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, cfg: DataConfig, files: Iterable[dict], outputs: Iterable[Path]) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg.name,
        "adapter": cfg.adapter,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "source_files": list(files),
        "outputs": [
            {"path": str(x), "size": x.stat().st_size, "sha256": sha256_file(x)} for x in outputs
        ],
        "strict_hash_validation": all(x["hash_declared"] for x in files),
        "pid": os.getpid(),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
