from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import yaml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--continue-on-error",action="store_true")
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    failures=[]
    for seed in cfg["seeds"]:
        command=[sys.executable,"scripts/train.py","--config",args.config,"--seed",str(seed), "--skip-pretrain"]
        completed=subprocess.run(command,check=False)
        if completed.returncode: failures.append(seed)
        if completed.returncode and not args.continue_on_error: raise SystemExit(completed.returncode)
    if failures: raise SystemExit(f"failed seeds: {failures}")

if __name__=="__main__": main()
