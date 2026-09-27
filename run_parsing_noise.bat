@echo off
REM ============================================================
REM  run_parsing_noise.bat
REM
REM  Section 6.5 parsing-noise robustness grid:
REM    4 corruption kinds (replace / merge / split / unk)
REM    x 2 contamination protocols (test_only / all)
REM    x 2 seeds (42, 123)  = 16 runs on SSH
REM
REM  Each variant first materialises a corrupted cache, then trains the
REM  main model on it. Finished runs are skipped, so the grid can be
REM  interrupted and restarted at any time.
REM
REM  Run it in a terminal OUTSIDE Codex, after the sensitivity sweep.
REM ============================================================

cd /d E:\SDHTG\sdhtg
set PY=D:\anaconda3\envs\sdhtg\python.exe

echo ############################################################
echo  parsing-noise grid (16 runs, SSH, rate=0.2)
echo ############################################################
%PY% scripts\run_parsing_noise.py --datasets ssh --seeds 42 123 --rate 0.2

echo PARSING-NOISE GRID DONE
echo   results: outputs\ssh\main\noise_r0.2\<kind>_<protocol>\seed_<s>\
pause
