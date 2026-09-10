@echo off
REM Standalone launcher for the P1 ablation groups (shortcuts / imbalance / boundary).
REM Run in a terminal OUTSIDE Codex so it survives across Codex sessions.
REM Resumes automatically (finished runs are skipped).

cd /d E:\SDHTG\sdhtg

echo ########## shortcuts ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group shortcuts --datasets hdfs ssh --seeds 42 123 256 512 1024

echo ########## imbalance ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group imbalance --datasets hdfs --seeds 42 123 256

echo ########## boundary ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group boundary --datasets hdfs ssh --seeds 42 123 256 512 1024

echo ALL ABLATIONS DONE
pause
