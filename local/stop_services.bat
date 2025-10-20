@echo off
REM AsterAI Services Stop Script

echo ==========================================
echo Stopping AsterAI Services
echo ==========================================

echo.
echo Stopping all Python processes...
taskkill /f /im python.exe 2>nul

echo.
echo ✅ All services stopped.
echo.
echo To restart services: Run 'start_services.bat'
echo ==========================================
pause
