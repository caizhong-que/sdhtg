@echo off
REM ============================================================
REM  run_scores_for_curves.bat
REM
REM  Dump per-sample test scores for the PR / threshold-curve
REM  figure: SSH (5 seeds, full test set) and HDFS (2 seeds,
REM  20000-sample stratified subsample) for the three unified
REM  protocol baselines and the flat GRU baseline.
REM
REM  Sequential by construction (one inference job at a time).
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe
set PYTHONIOENCODING=utf-8

for %%T in (baseline_tcn baseline_transformer baseline_gnn_flat ladder_full/L0) do (
  echo == ssh %%T
  %PY% scripts\dump_test_scores.py --dataset ssh --tag "%%T" --seeds 42 123 256 512 1024
  echo == hdfs %%T
  %PY% scripts\dump_test_scores.py --dataset hdfs --tag "%%T" --seeds 42 123 --max-samples 20000
)

echo SCORE DUMPS DONE
pause
