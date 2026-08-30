@echo off
REM Standalone launcher for the 5-dataset x 5-seed full ladder.
REM Run this in a terminal OUTSIDE Codex so it survives across sessions:
REM     call run_full_ladders.bat
REM It resumes automatically (--skip-existing) if interrupted.

cd /d E:\SDHTG\sdhtg

for %%D in (openstack bgl hdfs thunderbird) do (
    echo ########## DATASET %%D ##########
    D:\anaconda3\envs\sdhtg\python.exe scripts\run_ladder.py ^
        --config configs\experiment\%%D.yaml ^
        --levels L0,L1,L2,L3,L4,L5,L6,L7,L7b ^
        --seeds 42,123,256,512,1024 ^
        --max-epochs 30 ^
        --tag ladder_full ^
        --mask-template-prob 0.0 ^
        --skip-existing
    if errorlevel 1 (
        echo ERROR on %%D - see log above
        pause
        exit /b 1
    )
)

echo All ladders finished.
pause
