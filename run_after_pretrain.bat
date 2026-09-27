@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  run_after_pretrain.bat
REM
REM  Waits for the pretraining matrix, then runs everything that
REM  does not need the user in the loop:
REM
REM    0) wait until run_pretraining.py has exited
REM    1) reclaim last.pt of finished runs (frees ~38 GB)
REM    2) section 6.6 evidence: boundaries / pi / prototypes / cases
REM    3) prototype health of the reference model (K=8)
REM    4) section 6.8 sensitivity sweep (81 runs, skips finished ones)
REM
REM  Run it in a terminal OUTSIDE Codex. Every stage skips work that
REM  is already finished, so the script can be interrupted and
REM  restarted at any time.
REM
REM  Flags:  --skip-wait   do not wait for run_pretraining.py
REM          --dry-run     print the commands instead of running them
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe
set DRY=
set SKIPWAIT=
for %%A in (%*) do (
    if /I "%%A"=="--dry-run" set DRY=echo
    if /I "%%A"=="--skip-wait" set SKIPWAIT=1
)
if defined DRY echo [dry-run] commands are printed but not executed
if defined SKIPWAIT echo [skip-wait] not waiting for run_pretraining.py

if not defined SKIPWAIT (
    echo ############################################################
    echo  STEP 0  wait until the pretraining matrix has finished
    echo ############################################################
    %DRY% %PY% scripts\wait_for_pretraining.py ^
        --patterns run_pretraining.py --poll 300 --label pretraining
)

echo ############################################################
echo  STEP 1  reclaim last.pt of finished runs
echo ############################################################
%DRY% %PY% scripts\cleanup_outputs.py --apply

echo ############################################################
echo  STEP 2  section 6.6 evidence (boundaries / pi / prototypes / cases)
echo ############################################################
for %%S in (42 123 256) do (
    echo -- HDFS ladder_full/L7 seed %%S
    %DRY% %PY% scripts\interpret_boundaries.py --root outputs\hdfs\main\ladder_full\L7 --seed %%S --split test --device cuda --max-samples 1500
    %DRY% %PY% scripts\interpret_hierarchy.py  --root outputs\hdfs\main\ladder_full\L7 --seed %%S --split test --device cuda --max-samples 1500
    %DRY% %PY% scripts\interpret_prototypes.py --root outputs\hdfs\main\ladder_full\L7 --seed %%S --split test --device cuda --max-samples 1500
)
REM  Case selection needs a false positive AND a false negative. HDFS test has
REM  ~1.5% anomalies and the L7 model makes very few errors, so the scan covers
REM  the whole split (~3-5 min on GPU) instead of a prefix.
%DRY% %PY% scripts\interpret_cases.py --root outputs\hdfs\main\ladder_full\L7 --seed 42 --split test --device cuda --max-scan 200000 --top-k 8

for %%S in (42 123 256 512 1024) do (
    echo -- SSH ladder_full/L7 seed %%S
    %DRY% %PY% scripts\interpret_boundaries.py --root outputs\ssh\main\ladder_full\L7 --seed %%S --split test --device cuda
    %DRY% %PY% scripts\interpret_hierarchy.py  --root outputs\ssh\main\ladder_full\L7 --seed %%S --split test --device cuda
    %DRY% %PY% scripts\interpret_prototypes.py --root outputs\ssh\main\ladder_full\L7 --seed %%S --split test --device cuda
)
%DRY% %PY% scripts\interpret_cases.py --root outputs\ssh\main\ladder_full\L7 --seed 42 --split test --device cuda --top-k 8

echo ############################################################
echo  STEP 3  prototype health of the reference model (K=8)
echo ############################################################
for %%S in (42 123 256 512 1024) do (
    %DRY% %PY% scripts\prototype_stats.py --root outputs\ssh\main\ladder_full\L7 --seed %%S --split train --max-samples 5000 --device cuda
)

echo ############################################################
echo  STEP 4  section 6.8 sensitivity sweep (81 runs, skip-existing)
echo ############################################################
%DRY% %PY% scripts\run_sensitivity.py --datasets ssh --seeds 42 123 256 --prototype-stats

echo ############################################################
echo  STEP 5  backfill prototype health of runs whose stats were lost
echo ############################################################
%DRY% %PY% scripts\backfill_prototype_stats.py --datasets ssh --seeds 42 123 256

echo ALL POST-PRETRAIN STEPS DONE
echo   evidence     : outputs\<dataset>\main\ladder_full\L7\interpretability\
echo   sensitivity  : outputs\ssh\main\sens_<group>_<variant>\seed_<s>\
pause
