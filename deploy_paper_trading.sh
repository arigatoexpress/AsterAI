#!/bin/bash
# 🚀 Aster AI Paper Trading Deployment Script
# Generated: 2025-10-16 23:03:31

echo "🚀 Deploying Aster AI Paper Trading System to Cloud..."

# Set up environment
export PYTHONPATH=$PYTHONPATH:/app
cd /app

# Install dependencies
pip install -r requirements.txt

# Set up logging
mkdir -p /app/logs
mkdir -p /app/paper_trading_results

# Configure paper trading
export PAPER_TRADING_CAPITAL=10000.0
export MONITORING_INTERVAL=60

# Start paper trading system
python paper_trading_system.py &
echo "✅ Paper trading system started"

# Set up monitoring
python monitoring_system.py &
echo "✅ Monitoring system started"

# Set up backup system
crontab -l | { cat; echo "0 * * * * /app/backup_paper_trading.sh"; } | crontab -
echo "✅ Backup system configured"

echo "🎯 Paper trading deployment complete!"
echo "📊 Monitor at: http://localhost:8080"
echo "💾 Backups saved to: /app/paper_trading_results/"
