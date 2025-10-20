@echo off
REM Start local trading agent in background
REM Runs on local PC for ML training and GPU tasks

echo [START] Starting local AsterAI trading agent...

REM Check if already running
tasklist /FI "WINDOWTITLE eq AsterAI Local Trader" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [WARNING] Local trader already running
    exit /b 1
)

REM Check for virtual environment
if exist "asterai_venv\Scripts\activate.bat" (
    call asterai_venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [INFO] No virtual environment found, using system Python
)

REM Set environment variables for optimal local operation
set ASTERAI_MODE=local
set ASTERAI_ENABLE_GPU=1
set ASTERAI_FORCE_CPU=0
set PYTHONIOENCODING=utf-8

REM Set API keys from local config
if exist "local\.api_keys.json" (
    echo [OK] Using local API keys configuration
) else (
    echo [WARNING] No local API keys found
)

REM Start in new minimized window
echo [INFO] Starting dashboard on port 8081...
start "AsterAI Local Trader" /MIN python advanced_dashboard_server.py

REM Wait for startup
echo [INFO] Waiting for service to start...
timeout /t 5 /nobreak >nul

REM Check if running
curl -s http://localhost:8081/api/control/status >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [OK] Local trader started successfully
    echo [INFO] Dashboard: http://localhost:8081
    echo [INFO] Process running in background window
) else (
    echo [ERROR] Failed to start local trader
    echo [INFO] Check logs for details
    exit /b 1
)

echo.
echo [OK] Local trading agent is now running
echo [INFO] Use Ctrl+C in the background window to stop
echo.

exit /b 0

