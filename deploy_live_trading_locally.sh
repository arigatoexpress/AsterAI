#!/bin/bash
# 🚀 Aster AI Live Trading Bot Local Deployment Script
# Generated: 2025-10-20

echo "🚀 Deploying Aster AI Live Trading Bot Locally..."

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd "$(dirname "$0")"

# Check if virtual environment exists and activate it
if [ -d "asterai_env" ]; then
    echo "🔧 Activating virtual environment..."
    source asterai_env/Scripts/activate 2>/dev/null || source asterai_env/bin/activate 2>/dev/null
elif [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/Scripts/activate 2>/dev/null || source venv/bin/activate 2>/dev/null
fi

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p logs
mkdir -p live_trading_results
mkdir -p data/local_cache

# Set up logging
export LOG_LEVEL=INFO
export LOG_FILE=logs/live_trading_$(date +%Y%m%d_%H%M%S).log

# Configure live trading parameters
export LIVE_TRADING_CAPITAL=100.0
export DRY_RUN_MODE=true  # Start in dry-run mode for safety
export MAX_POSITIONS=2
export STOP_LOSS_PCT=0.02
export TAKE_PROFIT_PCT=0.04

echo "⚙️  Configuration:"
echo "   Capital: $LIVE_TRADING_CAPITAL"
echo "   Dry Run: $DRY_RUN_MODE"
echo "   Max Positions: $MAX_POSITIONS"
echo "   Stop Loss: $STOP_LOSS_PCT"
echo "   Take Profit: $TAKE_PROFIT_PCT"

# Check if Aster API credentials are configured
if [ ! -f ".api_keys.json" ]; then
    echo "⚠️  Warning: .api_keys.json not found!"
    echo "   Please configure your Aster API credentials first."
    echo "   The bot will run in simulation mode only."
    export SIMULATION_MODE=true
else
    export SIMULATION_MODE=false
    echo "✅ API credentials found"
fi

# Start live trading bot in background
echo "🤖 Starting Live Trading Bot..."
python live_trading_agent.py &
TRADING_PID=$!
echo "✅ Live trading bot started (PID: $TRADING_PID)"

# Start monitoring dashboard if available
if [ -f "dashboard_server.py" ]; then
    echo "📊 Starting monitoring dashboard..."
    python dashboard_server.py &
    DASHBOARD_PID=$!
    echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
    echo "   Monitor at: http://localhost:8081"
fi

# Set up signal handling for clean shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down live trading system..."
    if [ ! -z "$TRADING_PID" ]; then
        kill $TRADING_PID 2>/dev/null
        echo "✅ Trading bot stopped"
    fi
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null
        echo "✅ Dashboard stopped"
    fi
    echo "💾 System shutdown complete"
    exit 0
}

# Trap signals for clean shutdown
trap cleanup SIGINT SIGTERM

echo ""
echo "🎯 Live Trading System Deployed Successfully!"
echo ""
echo "📊 System Status:"
echo "   • Live Trading Bot: Running (PID: $TRADING_PID)"
if [ ! -z "$DASHBOARD_PID" ]; then
    echo "   • Monitoring Dashboard: http://localhost:8081 (PID: $DASHBOARD_PID)"
fi
echo "   • Log File: $LOG_FILE"
echo "   • Results Directory: live_trading_results/"
echo ""
echo "🛡️  Safety Features:"
echo "   • Dry-run mode enabled for initial testing"
echo "   • Risk management with stop-loss protection"
echo "   • Position size limits"
echo "   • Emergency stop capability"
echo ""
echo "⚠️  Important Notes:"
echo "   • Monitor the log file for trading activity"
echo "   • Check dashboard for real-time performance"
echo "   • Use Ctrl+C to stop the system gracefully"
echo ""
echo "🚀 System is running... Press Ctrl+C to stop"

# Wait for processes
wait
