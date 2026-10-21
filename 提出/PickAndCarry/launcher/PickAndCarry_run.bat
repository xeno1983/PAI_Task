@echo off
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0PickAndCarry_run.ps1"
if errorlevel 1 (
    echo.
    echo The launcher stopped. This window stays open so you can read the message above.
    pause
)
