@echo off
REM ============================================================
REM  RUN-FIRST: efficiency benchmark only (paper Table 11 / RQ6)
REM  No training is involved, so this finishes in ~1-2 hours and
REM  can safely run while other experiments occupy the GPU
REM  (it uses only forward/backward probes on synthetic batches).
REM ============================================================

cd /d E:\SDHTG\sdhtg

echo ========== efficiency benchmark: SDHTG vs baselines ==========
D:\anaconda3\envs\sdhtg\python.exe scripts\benchmark_efficiency_v2.py ^
    --models sdhtg gru_flat tcn transformer gnn_flat ^
    --lengths 32 64 128 256 512 ^
    --batch-sizes 512 128 64 64 64 ^
    --warmup 2 --repeats 5 ^
    --out outputs\efficiency_benchmark.json ^
    --skip-existing

echo.
echo Table 11 data written to outputs\efficiency_benchmark.json
echo Re-running this script only measures missing (model, length) pairs.
pause
