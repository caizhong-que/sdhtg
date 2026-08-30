@echo off
REM Unattended variant (no pause) for scheduled/detached execution.
REM Logs to full_ladders.log in the project root.

cd /d E:\SDHTG\sdhtg

for %%D in (openstack bgl hdfs thunderbird) do (
    echo ########## DATASET %%D %date% %time% ########## >> full_ladders.log
    D:\anaconda3\envs\sdhtg\python.exe scripts\run_ladder.py ^
        --config configs\experiment\%%D.yaml ^
        --levels L0,L1,L2,L3,L4,L5,L6,L7,L7b ^
        --seeds 42,123,256,512,1024 ^
        --max-epochs 30 ^
        --tag ladder_full ^
        --mask-template-prob 0.0 ^
        --skip-existing >> full_ladders.log 2>&1
)

echo ALL DONE %date% %time% >> full_ladders.log
