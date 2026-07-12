from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from sdhtg.data.config import load_config
from sdhtg.data.integrity import sha256_file


def download(url: str, destination: Path, expected_hash: str | None, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        if expected_hash and sha256_file(destination) != expected_hash.lower():
            raise ValueError(f"existing file hash mismatch: {destination}")
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(30, 300), allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary.open("wb") as stream, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    stream.write(chunk); digest.update(chunk); bar.update(len(chunk))
    actual = digest.hexdigest()
    if expected_hash and actual.lower() != expected_hash.lower():
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded hash mismatch: expected {expected_hash}, got {actual}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for spec in cfg.files:
        if not spec.url:
            print(f"manual download required for {spec.role}: {cfg.raw_dir / spec.path}")
            continue
        download(spec.url, cfg.raw_dir / spec.path, spec.sha256, args.overwrite)


if __name__ == "__main__":
    main()
