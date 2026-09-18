@echo off
REM Standalone launcher for the P1 ablation groups (shortcuts / imbalance / boundary).
REM Run in a terminal OUTSIDE Codex so it survives across Codex sessions.
REM Resumes automatically (finished runs are skipped).

cd /d E:\SDHTG\sdhtg

echo ########## shortcuts ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group shortcuts --datasets hdfs ssh --seeds 42 123 256 512 1024

echo ########## boundary (RQ2) ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group boundary --datasets hdfs ssh --seeds 42 123 256 512 1024

echo ########## candidate main models ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group candidates --datasets hdfs ssh --seeds 42 123 256 512 1024

echo ########## imbalance ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group imbalance --datasets hdfs ssh --seeds 42 123 256
REM 极端不平衡场景可追加 Thunderbird（较慢）：
REM D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
REM     --group imbalance --datasets thunderbird --seeds 42 123 256

echo ALL ABLATIONS DONE
pause
