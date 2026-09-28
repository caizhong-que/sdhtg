@echo off
REM ============================================================
REM  run_entity_holdout.bat
REM
REM  Section 6.5 seen/unseen entity grouping test.
REM
REM  The natural grouping is degenerate (SSH/OpenStack 0 unseen
REM  test sessions, BGL 0.28%, Thunderbird 0.30%, HDFS 100%),
REM  so the grouping is designed instead: the entity identities
REM  that cover ~20% of the test sessions are masked to UNK in
REM  train/validation/test, and the model is retrained.
REM
REM    bgl        20% coverage, seeds 42 123   (~8 min/run)
REM    openstack  20% coverage, seeds 42 123   (~1 min/run)
REM
REM  All stages run sequentially through scripts/run_entity_holdout.py
REM  (never two trainings at once: concurrent jobs exhaust the 8 GB
REM  card and fall back to shared memory).  Finished stages are
REM  skipped, so the script can be re-run at any time.
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe
set PYTHONIOENCODING=utf-8

echo ############################################################
echo  seen/unseen entity holdout: build cache, train, evaluate
echo ############################################################
%PY% scripts\run_entity_holdout.py --datasets bgl openstack --seeds 42 123 --max-seen 20000

echo ENTITY-HOLDOUT EXPERIMENT DONE
echo   group metrics: outputs\[dataset]\main\seen_unseen\entholdout20_L7_seed_[s].json
pause
