@echo off
REM ============================================================
REM  run_remaining.bat -- the three run sets still outstanding
REM
REM    1) epsilon_m = 1e-2 sensitivity cell -- SKIPPED BY DESIGN
REM       At epsilon_m >= 1e-4 the membership matrix stops being sparse
REM       (non-zero share 0.2% -> 50%, containment edges x500), so the
REM       standard batch does not fit in 8 GB and all three seeds fail with
REM       CUDA OOM. The cell is reported as "exceeds the memory budget".
REM    2) Table 9 backfill: CB-Focal + 8 prototypes (3 runs)
REM    3) Parsing-noise robustness grid (16 runs, rate 0.2)
REM
REM  Every step skips finished runs, so it can be interrupted and restarted.
REM  Run it in a terminal OUTSIDE Codex to keep it alive between turns.
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ############################################################
echo  [1/3]  epsilon_m = 1e-2 -- skipped (exceeds the memory budget)
echo ############################################################
echo    measured: membership non-zero 0.2%% -^> 50%%, containment edges 40 -^> 20300
echo    all three seeds fail with CUDA OOM at the standard batch size, so the
echo    cell is reported as infeasible instead of retried.

echo ############################################################
echo  [2/3]  Table 9 backfill: CB-Focal + 8 prototypes
echo ############################################################
%PY% scripts\run_ablations.py --group imbalance --datasets ssh ^
    --variants multi_prototype --seeds 42 123 256 --include-reference
if errorlevel 1 echo !! table 9 backfill reported a failure

echo ############################################################
echo  [3/3]  parsing-noise grid (16 runs)
echo ############################################################
%PY% scripts\run_parsing_noise.py --datasets ssh --seeds 42 123 --rate 0.2
if errorlevel 1 echo !! parsing-noise grid reported a failure

echo ALL REMAINING RUNS DONE
pause
