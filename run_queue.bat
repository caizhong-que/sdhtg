@echo off
REM ============================================================
REM  run_queue.bat -- remaining GPU work, strictly sequential
REM
REM    1. BGL temporal-split cache (CPU only)
REM    2. wait until no other training job is running
REM    3. seen/unseen entity holdout (bgl + openstack)
REM    4. BGL temporal split L0 vs L7 (section 6.5 promise)
REM    5. SSH label scarcity 1/5/10% for L0 and L7 (section 6.4)
REM    6. SSH parsing noise at 40% rewrite (section 6.5)
REM    7. per-sample scores for the PR-curve figure
REM
REM  Finished stages are skipped, so this can be re-run at any
REM  time (also after a reboot); it never starts two trainings at
REM  once, which is what made the 2026-09-28 runs 10x slower.
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe
set PYTHONIOENCODING=utf-8

%PY% scripts\run_queue.py

echo QUEUE DONE
pause
