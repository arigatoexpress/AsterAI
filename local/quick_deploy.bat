@echo off
REM AsterAI Quick Deployment Script for Windows

echo ==========================================
echo AsterAI Quick Deployment
echo ==========================================

echo.
echo [1/4] Starting Trading Server...
start /B python trading_server.py
timeout /t 2 /nobreak > nul

echo [2/4] Starting Dashboard Server...
start /B python dashboard_server.py
timeout /t 2 /nobreak > nul

echo [3/4] Starting Self-Learning Trader...
start /B python self_learning_trader.py
timeout /t 2 /nobreak > nul

echo [4/4] Verifying Deployment...
python verify_deployment.py

echo.
echo ==========================================
echo DEPLOYMENT COMPLETE
echo ==========================================
echo Trading Dashboard: http://localhost:8000
echo Trading Server: http://localhost:8001
echo.
echo Monitor logs in: logs\
echo View reports in: trading_analysis_reports\
echo.
echo To view live trading, open your browser to:
echo http://localhost:8000
echo ==========================================
pause
