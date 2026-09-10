@echo off
REM Standalone launcher for the unified-input baselines (TCN/Transformer/GNN-flat).
REM Run in a terminal OUTSIDE Codex so it survives across Codex sessions:
REM     call run_baselines.bat
REM Resumes automatically (already-finished runs are skipped).

cd /d E:\SDHTG\sdhtg

D:\anaconda3\envs\sdhtg\python.exe scripts\run_baselines.py ^
    --datasets openstack ssh bgl hdfs thunderbird ^
    --models tcn transformer gnn_flat ^
    --seeds 42 123 256 512 1024 ^
    --max-epochs 30

echo ALL BASELINES DONE
pause
