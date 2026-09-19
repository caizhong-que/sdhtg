@echo off
REM Standalone launcher for the P1 ablation groups (shortcuts / boundary /
REM imbalance / candidate main models).
REM
REM Run in a terminal OUTSIDE Codex so it survives across Codex sessions.
REM Every run whose result.json already exists is SKIPPED, so this script is
REM safe to re-run and never repeats finished experiments.
REM
REM Already finished (NOT listed below, will not be repeated):
REM   * ladders on all 5 datasets (225 runs)
REM   * unified-input baselines TCN / Transformer / GNN-flat (75 runs)
REM   * HDFS shortcuts: no_entity_embed, no_entity_prior, no_action_prior
REM   * label-scarcity runs on bgl_iter (1%/5%/10% x {L0,L7})
REM
REM Strategy: SSH is cheap (~15 min/run) and carries the COMPLETE ablation
REM tables; HDFS is expensive (~3.5 h/run) and carries only the variants the
REM paper's claims need.

cd /d E:\SDHTG\sdhtg

echo ========== PHASE A1: SSH shortcuts (full 9 variants) ==========
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group shortcuts --datasets ssh --seeds 42 123 256 512 1024

echo ========== PHASE A2: SSH boundary / RQ2 (full 6 variants) ==========
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group boundary --datasets ssh --seeds 42 123 256 512 1024

echo ========== PHASE A3: SSH imbalance (loss / prototype) ==========
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group imbalance --datasets ssh --seeds 42 123 256

echo ========== PHASE A4: SSH candidate main model ==========
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group candidates --datasets ssh --seeds 42 123 256 512 1024

echo ========== PHASE B1: HDFS candidate main model ==========
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group candidates --datasets hdfs --seeds 42 123 256 512 1024

echo ========== PHASE B2: HDFS shortcuts (key variants only) ==========
REM entity_unk and shuffle_entity cover the entity-shortcut claim; the three
REM content shortcuts (status source / explicit words / template-only) carry
REM the "no shortcut reliance" claim. no_action_source is skipped because
REM no_action_prior is already measured on HDFS.
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group shortcuts --datasets hdfs ^
    --variants shuffle_entity entity_unk no_status_source mask_status_words template_time_only ^
    --seeds 42 123 256 512 1024

echo ========== PHASE B3: HDFS boundary (key variants only) ==========
REM detach = cross-level gradient path; fixed_window = learned vs preset.
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group boundary --datasets hdfs --variants detach fixed_window ^
    --seeds 42 123 256 512 1024

REM ========== OPTIONAL (uncomment when needed) ==========
REM HDFS imbalance (slow, ~2.6 days):
REM D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
REM     --group imbalance --datasets hdfs --seeds 42 123 256
REM Thunderbird imbalance (extreme imbalance, ~1.5 days):
REM D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
REM     --group imbalance --datasets thunderbird --seeds 42 123 256
REM HDFS full shortcuts / boundary (if extra time is available):
REM D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
REM     --group shortcuts --datasets hdfs --seeds 42 123 256 512 1024
REM D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
REM     --group boundary --datasets hdfs --seeds 42 123 256 512 1024

echo ALL ABLATIONS DONE
pause
