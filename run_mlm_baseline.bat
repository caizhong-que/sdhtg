@echo off
REM ============================================================
REM  run_mlm_baseline.bat
REM
REM  Masked-template pretraining baseline (LogBERT route) under
REM  the unified input protocol:
REM
REM    ssh         5 seeds  (dynamic-range dataset)
REM    openstack   5 seeds  (small, cheap)
REM    bgl         1 seed   (saturated dataset; kept for completeness)
REM
REM  Protocol matches the main ladder: 10 masked-template pretraining
REM  epochs, up to 25 supervised epochs, early stopping on validation
REM  AUPRC (patience 10), threshold calibrated on validation by F1.
REM  Finished runs are skipped, so this can be restarted at any time.
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe
set PYTHONIOENCODING=utf-8

%PY% scripts\train_mlm_baseline.py --dataset ssh --seeds 42 123 256 512 1024
%PY% scripts\train_mlm_baseline.py --dataset openstack --seeds 42 123 256 512 1024
%PY% scripts\train_mlm_baseline.py --dataset bgl --seeds 42

echo MLM BASELINE DONE
pause
