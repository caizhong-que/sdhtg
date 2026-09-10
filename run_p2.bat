@echo off
REM Standalone launcher for the P2 experiments:
REM   1) efficiency benchmark (Table 11)
REM   2) contrastive-pretraining matrix (Table 10)
REM Run in a terminal OUTSIDE Codex so it survives across Codex sessions.

cd /d E:\SDHTG\sdhtg

echo ########## efficiency benchmark ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\benchmark_efficiency_v2.py ^
    --models sdhtg gru_flat tcn transformer gnn_flat ^
    --lengths 32,64,128,256,512 --batch-sizes 512,128,64,64,64

echo ########## pretraining matrix ##########
D:\anaconda3\envs\sdhtg\python.exe scripts\run_pretraining.py ^
    --datasets ssh --protocols normal_only all_train ^
    --strategies random hard semi_hard semantic none supervised ^
    --seeds 42 123 256 --pretrain-epochs 10

echo ALL P2 EXPERIMENTS DONE
pause
