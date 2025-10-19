#!/usr/bin/env python3
"""
Simple HTTP server wrapper for the autonomous trading agent.
Provides health checks and basic monitoring endpoints.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Aster Trading Agent", version="1.0.0")

# Global state
trading_agent = None
is_running = False

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "aster-trading-agent",
        "running": is_running
    }

@app.get("/status")
async def get_status():
    """Get trading agent status."""
    if not trading_agent:
        return {"status": "not_initialized"}
    
    return {
        "status": "running" if is_running else "stopped",
        "timestamp": datetime.now().isoformat(),
        "agent_type": "autonomous_mcp_agent"
    }

@app.post("/start")
async def start_trading():
    """Start the trading agent."""
    global is_running, trading_agent
    
    if is_running:
        return {"status": "already_running"}
    
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
            
            # Initialize Aster client with API keys from environment
            api_key = os.getenv("ASTER_API_KEY")
            secret_key = os.getenv("ASTER_API_SECRET")
            
            if not api_key or not secret_key:
                raise ValueError("ASTER_API_KEY and ASTER_API_SECRET environment variables must be set")
            
            aster_client = AsterClient(api_key=api_key, secret_key=secret_key)
            
            # Create and start the live trading agent
            trading_agent = LiveTradingAgent(config, aster_client)
            
            # Start the agent in the background
            asyncio.create_task(trading_agent.start_trading())
            is_running = True
            
            logger.info("Fallback trading agent started successfully")
            return {"status": "started", "message": "Fallback trading agent is now running"}
        
    except Exception as e:
        logger.error(f"Failed to start trading agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start trading agent: {str(e)}")

@app.post("/stop")
async def stop_trading():
    """Stop the trading agent."""
    global is_running, trading_agent
    
    if not is_running:
        return {"status": "not_running"}
    
    try:
        if trading_agent:
            await trading_agent.stop()
        is_running = False
        
        logger.info("Trading agent stopped")
        return {"status": "stopped", "message": "Trading agent has been stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop trading agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop trading agent: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Aster Trading Agent",
        "version": "1.0.0",
        "status": "running" if is_running else "stopped",
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "start": "POST /start",
            "stop": "POST /stop"
        }
    }

@app.get("/debug")
async def debug_env():
    """Debug endpoint to check environment variables."""
    return {
        "environment_variables": {
            "ASTER_API_KEY": "SET" if os.getenv("ASTER_API_KEY") else "NOT_SET",
            "ASTER_API_SECRET": "SET" if os.getenv("ASTER_API_SECRET") else "NOT_SET",
            "ENVIRONMENT": os.getenv("ENVIRONMENT"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL")
        },
        "all_env_vars": {k: v for k, v in os.environ.items() if "ASTER" in k or "ENVIRONMENT" in k or "LOG" in k}
    }

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting Aster Trading Agent server on port {port}")
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
