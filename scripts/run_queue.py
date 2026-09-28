# -*- coding: utf-8 -*-
"""run_queue.py -- sequential queue for the remaining GPU work.

Stages (each skips finished artefacts, so the queue can be restarted at any
time, e.g. after a reboot):

    1. preprocess   rebuild the BGL temporal-split cache (CPU only)
    2. wait         block until no other training job is running
    3. holdout      seen/unseen entity study (bgl + openstack)
    4. temporal     BGL temporal split, L0 vs L7, 2 seeds
    5. scarcity     label scarcity on SSH, L0 vs L7, 1/5/10%, 3 seeds
    6. noise40      parsing noise on SSH at 40% rewrite, 4 kinds x 2 protocols
    7. scores       per-sample scores for the PR-curve figure

Every training job runs sequentially: two concurrent jobs exhaust the 8 GB card
and fall back to shared memory, which costs roughly 10x per step.

Usage:
    python scripts/run_queue.py --dry-run
    python scripts/run_queue.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


JOB_SCRIPTS = (
    "train.py",
    "run_entity_holdout.py",
    "run_label_scarcity.py",
    "run_ladder.py",
    "run_parsing_noise.py",
    "dump_test_scores.py",
    "evaluate_seen_unseen.py",
    "preprocess.py",
)


def training_running() -> bool:
    """True only when another *python* job is alive.

    Wrapper processes must be ignored: a finished ``run_entity_holdout.bat``
    stays alive at its ``pause`` prompt, and counting its command line as "still
    training" made the queue wait forever on 2026-09-28.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
    if psutil is not None:
        for process in psutil.process_iter(["name", "cmdline"]):
            name = (process.info.get("name") or "").lower()
            if not name.startswith("python"):
                continue
            if process.pid == os.getpid():
                continue
            cmdline = " ".join(process.info.get("cmdline") or []).replace("\\", "/")
            if any(script in cmdline for script in JOB_SCRIPTS):
                return True
        return False
    output = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                            capture_output=True, text=True).stdout
    return output.lower().count("python.exe") > 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=21600)
    parser.add_argument("--temporal-epochs", type=int, default=8)
    parser.add_argument("--scarcity-epochs", type=int, default=25)
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")

    def run(command: list[str], label: str) -> int:
        print(f"[queue] {label}: {' '.join(command)}", flush=True)
        if args.dry_run:
            return 0
        return subprocess.call(command, env=environment)

    # ---------------------------------------------------------------- stage 1
    print("[queue] stage 1: BGL temporal cache", flush=True)
    if Path("data/processed/bgl_temporal/sessions.parquet").is_file():
        print("[queue]   cache exists")
    else:
        run([sys.executable, "scripts/preprocess.py",
             "--config", "configs/data/bgl_temporal.yaml"], "preprocess")

    # ---------------------------------------------------------------- stage 2
    print("[queue] stage 2: waiting for free GPU", flush=True)
    if not args.dry_run:
        started = time.time()
        while training_running():
            if time.time() - started > args.wait_timeout:
                raise SystemExit("[queue] wait timeout")
            time.sleep(60)
        print("[queue]   GPU free")

    # ---------------------------------------------------------------- stage 3
    holdout_done = all(
        Path(f"outputs/{ds}/main/seen_unseen/entholdout20_L7_seed_{seed}.json").is_file()
        for ds in ("bgl", "openstack") for seed in (42, 123)
    )
    if args.skip_holdout or holdout_done:
        print("[queue] stage 3: entity holdout already done")
    else:
        print("[queue] stage 3: seen/unseen entity holdout", flush=True)
        run([sys.executable, "scripts/run_entity_holdout.py",
             "--datasets", "bgl", "openstack", "--seeds", "42", "123",
             "--max-seen", "20000"], "holdout")

    # ---------------------------------------------------------------- stage 4
    temporal_done = all(
        Path(f"outputs/bgl_temporal/main/temporal/{level}/seed_{seed}/result.json").is_file()
        for level in ("L0", "L7") for seed in (42, 123)
    )
    if temporal_done:
        print("[queue] stage 4: temporal split already done")
    else:
        print("[queue] stage 4: BGL temporal split (L0 vs L7)", flush=True)
        run([sys.executable, "scripts/run_ladder.py",
             "--config", "configs/experiment/bgl_temporal.yaml",
             "--levels", "L0,L7", "--seeds", "42,123",
             "--max-epochs", str(args.temporal_epochs),
             "--pretrain-epochs", "0", "--mask-template-prob", "0.0",
             "--tag", "temporal", "--skip-existing"], "temporal")

    # ---------------------------------------------------------------- stage 5
    print("[queue] stage 5: SSH label scarcity", flush=True)
    run([sys.executable, "scripts/run_label_scarcity.py",
         "--dataset", "ssh", "--levels", "L0", "L7",
         "--fractions", "0.01", "0.05", "0.10",
         "--seeds", "42", "123", "256",
         "--max-epochs", str(args.scarcity_epochs)], "scarcity")

    # ---------------------------------------------------------------- stage 6
    print("[queue] stage 6: SSH parsing noise at 40%", flush=True)
    run([sys.executable, "scripts/run_parsing_noise.py",
         "--datasets", "ssh", "--seeds", "42", "123", "--rate", "0.4"], "noise40")

    # ---------------------------------------------------------------- stage 7
    print("[queue] stage 7: per-sample scores for PR curves", flush=True)
    run([sys.executable, "scripts/dump_test_scores.py", "--dataset", "ssh",
         "--level", "L7", "--seeds", "42", "123", "256", "512", "1024"], "scores-ssh")
    run([sys.executable, "scripts/dump_test_scores.py", "--dataset", "hdfs",
         "--level", "L7", "--seeds", "42", "123", "--max-samples", "20000"], "scores-hdfs")

    print("[queue] all stages finished" if not args.dry_run
          else "[queue] dry run only")


if __name__ == "__main__":
    main()
