@echo off
REM ============================================================
REM  run_remaining.bat -- the three run sets still outstanding
REM
REM    1) epsilon_m = 1e-2 sensitivity cell (3 runs)
REM       The hierarchy saturates at this setting (one action/entity node per
REM       event, graph edges roughly double), so the batch is reduced to 256.
REM    2) Table 9 backfill: CB-Focal + 8 prototypes (3 runs)
REM    3) Parsing-noise robustness grid (16 runs, rate 0.2)
REM
REM  Every step skips finished runs, so it can be interrupted and restarted.
REM  Run it in a terminal OUTSIDE Codex to keep it alive between turns.
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ############################################################
echo  [1/3]  epsilon_m = 1e-2  (batch reduced to 256)
echo ############################################################
for %%S in (42 123 256) do (
    if not exist "outputs\ssh\main\sens_membership_epsilon_eps1e-2\seed_%%S\result.json" (
        echo -- seed %%S
        %PY% scripts\train.py --config configs\experiment\ssh.yaml ^
            --model-config configs\model\sdhtg.yaml ^
            --seed %%S --tag sens_membership_epsilon_eps1e-2 ^
            --max-epochs 30 --pretrain-epochs 0 --skip-pretrain ^
            --mask-template-prob 0.0 ^
            --model-set hierarchy.membership_epsilon=1.0e-2 ^
            --set batch_size=256
    )
)

echo ############################################################
echo  [2/3]  Table 9 backfill: CB-Focal + 8 prototypes
echo ############################################################
%PY% scripts\run_ablations.py --group imbalance --datasets ssh ^
    --variants multi_prototype --seeds 42 123 256 --include-reference

echo ############################################################
echo  [3/3]  parsing-noise grid (16 runs)
echo ############################################################
%PY% scripts\run_parsing_noise.py --datasets ssh --seeds 42 123 --rate 0.2

echo ALL REMAINING RUNS DONE
pause
