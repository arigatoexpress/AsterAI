@echo off
REM AsterAI Services Startup Script

echo ==========================================
echo AsterAI Services Startup
echo ==========================================

echo.
echo Starting services on different ports to avoid conflicts...
echo.

echo [1/3] Starting Dashboard Server (port 8081)...
start /B python dashboard_server.py
timeout /t 3 /nobreak > nul

echo [2/3] Starting Trading Server (port 8001)...
start /B cmd /c "$env:PORT=8001 && python enhanced_trading_server.py"
timeout /t 3 /nobreak > nul

echo [3/3] Cloud bot is already running in Google Cloud Run
echo.

echo ==========================================
echo SERVICES STARTED SUCCESSFULLY!
echo ==========================================
echo.
echo 🌐 Access URLs:
echo    📊 Dashboard:    http://localhost:8081
echo    🤖 Trading API:  http://localhost:8001/health
echo    ☁️  Cloud Bot:    https://aster-self-learning-trader-880429861698.us-central1.run.app/status
echo.
echo 📋 Status Report: SERVICES_STATUS.md
echo.
echo ⚠️  To stop services: Run 'stop_services.bat'
echo ==========================================
pause
