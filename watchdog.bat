@echo off
REM ============================================================
REM  watchdog.bat -- keep the experiment queue running across reboots
REM
REM  The machine has been losing power under sustained GPU load (six hard
REM  power-offs in two days), which kills whatever was training. This script
REM  loops forever: whenever no queue process is alive, it relaunches the
REM  outstanding work. Every runner skips finished runs, so relaunching is
REM  always safe and never repeats completed experiments.
REM
REM  Register it to start automatically at logon (run once, in an admin shell):
REM      schtasks /create /tn "SDHTG-Queue" /tr "E:\SDHTG\sdhtg\watchdog.bat" ^
REM          /sc onlogon /rl highest /f
REM  Remove it with:
REM      schtasks /delete /tn "SDHTG-Queue" /f
REM
REM  Or simply call it manually after a reboot.
REM ============================================================

cd /d E:\SDHTG\sdhtg

:loop
tasklist /fi "imagename eq python.exe" | find /i "python.exe" >nul
if errorlevel 1 (
    echo [%date% %time%] no training process found - starting the queue
    call run_remaining.bat
) else (
    echo [%date% %time%] training in progress - waiting
)
timeout /t 300 /nobreak >nul
goto loop
