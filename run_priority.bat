@echo off
REM ============================================================
REM  PRIORITY-ORDERED RUN
REM    1) efficiency benchmark          (Table 11, no training)
REM    2) candidate main models         (decide the final model)
REM    3) SSH ablations                 (cheap, complete tables)
REM    4) HDFS key variants             (expensive, minimal set)
REM    5) pretraining matrix            (Table 10)
REM    6) domain baselines (DeepLog/LogAnomaly/LogBERT, original protocol)
REM
REM  Every step skips finished runs, so the script can be interrupted
REM  and re-started at any time without repeating experiments.
REM  Run it in a terminal OUTSIDE Codex to keep it alive between turns.
REM ============================================================

cd /d E:\SDHTG\sdhtg

echo ############################################################
echo  STEP 1  efficiency benchmark (Table 11) - already finished,
echo          this call only fills any missing (model, length) pair
echo ############################################################
D:\anaconda3\envs\sdhtg\python.exe scripts\benchmark_efficiency_v2.py ^
    --models sdhtg gru_flat tcn transformer gnn_flat ^
    --lengths 32 64 128 256 512 --batch-sizes 512 128 64 64 64 ^
    --out outputs\efficiency_benchmark.json --skip-existing

echo ############################################################
echo  STEP 2  candidate main models (hierarchy+prototypes, no graph edges)
echo ############################################################
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group candidates --datasets ssh --seeds 42 123 256 512 1024
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group candidates --datasets hdfs --seeds 42 123 256 512 1024

echo ############################################################
echo  STEP 3  SSH ablations (shortcuts / boundary / imbalance)
echo ############################################################
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group shortcuts --datasets ssh --seeds 42 123 256 512 1024
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group boundary --datasets ssh --seeds 42 123 256 512 1024
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group imbalance --datasets ssh --seeds 42 123 256

echo ############################################################
echo  STEP 4  HDFS key variants (shortcuts / boundary)
echo ############################################################
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group shortcuts --datasets hdfs ^
    --variants shuffle_entity entity_unk no_status_source mask_status_words template_time_only ^
    --seeds 42 123 256 512 1024
D:\anaconda3\envs\sdhtg\python.exe scripts\run_ablations.py ^
    --group boundary --datasets hdfs --variants detach fixed_window ^
    --seeds 42 123 256 512 1024

echo ############################################################
echo  STEP 5  contrastive pretraining matrix (Table 10)
echo ############################################################
D:\anaconda3\envs\sdhtg\python.exe scripts\run_pretraining.py ^
    --datasets ssh --protocols normal_only all_train ^
    --strategies random hard semi_hard semantic none supervised ^
    --seeds 42 123 256 --pretrain-epochs 10

echo ############################################################
echo  STEP 6  domain baselines (original protocol, raw LogHub data)
echo          DeepLog / LogAnomaly / LogBERT via baselines/logbert
echo ############################################################
D:\anaconda3\envs\sdhtg\python.exe scripts\run_logbert_baselines.py ^
    --mode all --datasets bgl hdfs openstack ssh thunderbird ^
    --baselines deeplog loganomaly logbert ^
    --seeds 42 123 256 512 1024

echo ALL PRIORITY STEPS DONE
pause
