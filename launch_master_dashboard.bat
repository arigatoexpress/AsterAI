@echo off
echo Starting Aster AI Master Dashboard...
echo.
echo This will show the status of all your deployed services:
echo - Trading Bot
echo - Next.js Dashboard  
echo - GraphQL Gateway
echo - Trading Dashboard
echo - Rari Trade Dashboard
echo - Aster Dashboard
echo - Quant AI Trader
echo.
echo Dashboard will open at: http://localhost:8082
echo.
python -m http.server 8082
pause
