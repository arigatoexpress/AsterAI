#!/usr/bin/env python3
"""
Enhanced HTTP server wrapper for the autonomous trading agent.
Provides health checks, monitoring endpoints, and Telegram integration.
"""

import asyncio
import logging
import os
import signal
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Aster Trading Agent",
    version="1.0.0",
    description="AI-powered trading bot with real-time monitoring and Telegram integration"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trading_agent = None
is_running = False
trades_log: List[Dict[str, Any]] = []
performance_metrics = {
    "total_trades": 0,
    "total_pnl": 0.0,
    "win_rate": 0.0,
    "current_balance": 100.0,
    "open_positions": 0,
    "last_trade_time": None,
    "max_drawdown": 0.0,
    "sharpe_ratio": 0.0
}

# Telegram integration
telegram_enabled = False
telegram_notifier = None

def initialize_telegram():
    """Initialize Telegram integration if credentials are available"""
    global telegram_enabled, telegram_notifier
    
    try:
        from telegram_bot import initialize_telegram, send_trade_notification, send_error_notification, send_status_update
        
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if telegram_bot_token and telegram_chat_id:
            telegram_enabled = initialize_telegram(telegram_bot_token, telegram_chat_id)
            if telegram_enabled:
                logger.info("Telegram integration initialized successfully")
            else:
                logger.warning("Failed to initialize Telegram integration")
        else:
            logger.info("Telegram credentials not provided, notifications disabled")
            
    except ImportError:
        logger.warning("Telegram integration not available - telegram_bot module not found")

async def send_telegram_notification(notification_type: str, data: Dict[str, Any]):
    """Send notification via Telegram"""
    if not telegram_enabled:
        return
        
    try:
        from telegram_bot import send_trade_notification, send_error_notification, send_status_update
        
        if notification_type == "trade":
            await send_trade_notification(data)
        elif notification_type == "error":
            await send_error_notification(data.get("error", ""), data.get("context", ""))
        elif notification_type == "status":
            await send_status_update(data)
            
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")

def update_performance_metrics():
    """Update performance metrics based on trades log"""
    if not trades_log:
        return
        
    # Calculate total PnL
    total_pnl = sum(trade.get("pnl", 0) for trade in trades_log)
    performance_metrics["total_pnl"] = total_pnl
    performance_metrics["current_balance"] = 100.0 + total_pnl
    
    # Calculate win rate
    winning_trades = sum(1 for trade in trades_log if trade.get("pnl", 0) > 0)
    performance_metrics["win_rate"] = (winning_trades / len(trades_log)) * 100 if trades_log else 0
    
    # Update other metrics
    performance_metrics["total_trades"] = len(trades_log)
    if trades_log:
        performance_metrics["last_trade_time"] = trades_log[-1].get("ts", "")

def log_trade(trade_data: Dict[str, Any]):
    """Log a trade and update metrics"""
    trade_data["timestamp"] = datetime.now().isoformat()
    trades_log.append(trade_data)
    
    # Keep only last 100 trades
    if len(trades_log) > 100:
        trades_log.pop(0)
    
    update_performance_metrics()
    
    # Send Telegram notification
    if telegram_enabled:
        asyncio.create_task(send_telegram_notification("trade", trade_data))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "aster-trading-agent",
        "running": is_running,
        "telegram_enabled": telegram_enabled
    }

@app.get("/status")
async def get_status():
    """Get trading agent status with detailed information."""
    return {
        "status": "running" if is_running else "stopped",
        "timestamp": datetime.now().isoformat(),
        "agent_type": "autonomous_mcp_agent",
        "performance": performance_metrics,
        "telegram_enabled": telegram_enabled
    }

@app.get("/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    return {
        "trades": trades_log[-limit:] if trades_log else [],
        "total_count": len(trades_log),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    return {
        "metrics": performance_metrics,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/start")
async def start_trading(background_tasks: BackgroundTasks):
    """Start the trading agent."""
    global is_running, trading_agent
    
    if is_running:
        return {"status": "already_running", "message": "Trading agent is already running"}
    
    try:
        # Try to import and initialize the trading agent
        try:
            from autonomous_mcp_agent import AutonomousMCPAgent, MasterAgentConfig
            
            config = MasterAgentConfig(
                max_position_size_usd=100.0,  # $100 capital as requested
                max_daily_loss_pct=0.10,      # 10% max daily loss
                stop_loss_pct=0.02,           # 2% stop loss
                take_profit_pct=0.04,         # 4% take profit
            )
            
            trading_agent = AutonomousMCPAgent(config)
            
            # Start the agent in the background
            asyncio.create_task(trading_agent.run())
            is_running = True
            
            logger.info("Trading agent started successfully")
            
            # Send Telegram notification
            if telegram_enabled:
                background_tasks.add_task(
                    send_telegram_notification, 
                    "status", 
                    {"running": True, "message": "Trading bot started successfully"}
                )
            
            return {"status": "started", "message": "Trading agent is now running"}
            
        except ImportError as ie:
            # Fallback to simpler trading agent if MCP modules are missing
            logger.warning(f"MCP modules not available: {ie}, using fallback agent")
            
            from live_trading_agent import LiveTradingAgent, TradingConfig
            from mcp_trader.execution.aster_client import AsterClient
            
            # Create a simple trading config
            config = TradingConfig(
                initial_capital=100.0,
                position_size_pct=0.1,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                daily_loss_limit_pct=0.10
            )
            
            # Initialize Aster client (will use environment variables for API keys)
            aster_client = AsterClient()
            
            # Create and start the live trading agent
            trading_agent = LiveTradingAgent(config, aster_client)
            
            # Start the agent in the background
            asyncio.create_task(trading_agent.run())
            is_running = True
            
            logger.info("Fallback trading agent started successfully")
            
            # Send Telegram notification
            if telegram_enabled:
                background_tasks.add_task(
                    send_telegram_notification, 
                    "status", 
                    {"running": True, "message": "Fallback trading agent started successfully"}
                )
            
            return {"status": "started", "message": "Fallback trading agent is now running"}
        
    except Exception as e:
        logger.error(f"Failed to start trading agent: {e}")
        
        # Send error notification
        if telegram_enabled:
            background_tasks.add_task(
                send_telegram_notification, 
                "error", 
                {"error": str(e), "context": "Failed to start trading agent"}
            )
        
        raise HTTPException(status_code=500, detail=f"Failed to start trading agent: {str(e)}")

@app.post("/stop")
async def stop_trading(background_tasks: BackgroundTasks):
    """Stop the trading agent."""
    global is_running, trading_agent
    
    if not is_running:
        return {"status": "not_running", "message": "Trading agent is not running"}
    
    try:
        if trading_agent:
            await trading_agent.stop()
        is_running = False
        
        logger.info("Trading agent stopped")
        
        # Send Telegram notification
        if telegram_enabled:
            background_tasks.add_task(
                send_telegram_notification, 
                "status", 
                {"running": False, "message": "Trading bot stopped"}
            )
        
        return {"status": "stopped", "message": "Trading agent has been stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop trading agent: {e}")
        
        # Send error notification
        if telegram_enabled:
            background_tasks.add_task(
                send_telegram_notification, 
                "error", 
                {"error": str(e), "context": "Failed to stop trading agent"}
            )
        
        raise HTTPException(status_code=500, detail=f"Failed to stop trading agent: {str(e)}")

@app.post("/trade")
async def log_trade_endpoint(trade_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Log a trade (for testing or external integration)."""
    try:
        log_trade(trade_data)
        return {"status": "success", "message": "Trade logged successfully"}
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log trade: {str(e)}")

@app.get("/dashboard")
async def get_dashboard_data():
    """Get comprehensive dashboard data."""
    return {
        "status": {
            "running": is_running,
            "timestamp": datetime.now().isoformat(),
            "telegram_enabled": telegram_enabled
        },
        "performance": performance_metrics,
        "recent_trades": trades_log[-10:] if trades_log else [],
        "summary": {
            "total_trades": len(trades_log),
            "total_pnl": performance_metrics["total_pnl"],
            "win_rate": performance_metrics["win_rate"],
            "current_balance": performance_metrics["current_balance"]
        }
    }

@app.get("/")
async def root():
    """Root endpoint with comprehensive info."""
    return {
        "service": "Aster Trading Agent",
        "version": "1.0.0",
        "status": "running" if is_running else "stopped",
        "features": [
            "Real-time trading",
            "Telegram notifications",
            "Performance monitoring",
            "REST API",
            "Dashboard integration"
        ],
        "endpoints": {
            "health": "GET /health",
            "status": "GET /status",
            "trades": "GET /trades",
            "metrics": "GET /metrics",
            "dashboard": "GET /dashboard",
            "start": "POST /start",
            "stop": "POST /stop",
            "trade": "POST /trade"
        },
        "telegram_enabled": telegram_enabled
    }

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    # Initialize Telegram integration
    initialize_telegram()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting Enhanced Aster Trading Agent server on port {port}")
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
