@echo off
REM ============================================================
REM  P2 experiments, in execution order:
REM    STEP 1  efficiency benchmark          (Table 11, no training)
REM    STEP 2  contrastive pretraining matrix (Table 10, training)
REM
REM Run in a terminal OUTSIDE Codex so it survives across sessions.
REM Both steps skip work that is already finished, so the script is
REM safe to interrupt and re-run.
REM ============================================================

cd /d E:\SDHTG\sdhtg

echo ============================================================
echo  STEP 1/2  efficiency benchmark (Table 11) - no training
echo ============================================================
D:\anaconda3\envs\sdhtg\python.exe scripts\benchmark_efficiency_v2.py ^
    --models sdhtg gru_flat tcn transformer gnn_flat ^
    --lengths 32 64 128 256 512 ^
    --batch-sizes 512 128 64 64 64 ^
    --warmup 2 --repeats 5 ^
    --out outputs\efficiency_benchmark.json ^
    --skip-existing

echo.
echo ============================================================
echo  STEP 2/2  pretraining matrix (Table 10)
echo            2 protocols x 6 negative strategies x 3 seeds = 36 runs
echo            (each run = 10 pretrain epochs + fine-tuning)
echo ============================================================
D:\anaconda3\envs\sdhtg\python.exe scripts\run_pretraining.py ^
    --datasets ssh --protocols normal_only all_train ^
    --strategies random hard semi_hard semantic none supervised ^
    --seeds 42 123 256 --pretrain-epochs 10

REM ============ OPTIONAL (uncomment when needed) ============
REM Pretraining matrix on the second dynamic-range dataset:
REM D:\anaconda3\envs\sdhtg\python.exe scripts\run_pretraining.py ^
REM     --datasets hdfs --protocols normal_only all_train ^
REM     --strategies random hard semi_hard semantic none supervised ^
REM     --seeds 42 123
REM Pretraining + the candidate main model together:
REM D:\anaconda3\envs\sdhtg\python.exe scripts\run_pretraining.py ^
REM     --datasets ssh --model-config configs\model\cand_nograph.yaml ^
REM     --seeds 42 123 256

echo ALL P2 EXPERIMENTS DONE
pause
