#!/usr/bin/env python3
"""
FastAPI server for Self-Learning Aggressive Trading Bot
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from self_learning_trader import SelfLearningTrader, TradingConfig
    _self_learning_available = True
    logger.info("Self-learning trader loaded successfully")
except ImportError as e:
    logger.warning(f"Self-learning trader not available: {e}")
    _self_learning_available = False
    # Fallback to basic trading agent
    from live_trading_agent import LiveTradingAgent, TradingConfig as BasicTradingConfig
    logger.info("Using fallback basic trading agent")

# Initialize FastAPI app
app = FastAPI(title="Self-Learning Trading Bot", version="2.0.0")

# Global state
trading_bot = None
is_running = False
performance_data = []

# Cloud Service Controller
cloud_controller = None

def initialize_cloud_controller():
    """Initialize the cloud service controller"""
    global cloud_controller
    try:
        # Import the CloudServiceController directly
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

        # Copy the CloudServiceController class inline
        class CloudServiceController:
            """Controller for cloud trading services"""

            def __init__(self):
                # GCP Project details
                self.project_id = "quant-ai-trader-credits"
                self.region = "us-central1"
                self.project_num = "880429861698"

                # Service URLs
                self.services = {
                    "aster-trading-agent": f"https://aster-trading-agent-{self.project_num}.{self.region}.run.app",
                    "aster-self-learning-trader": f"https://aster-self-learning-trader-{self.project_num}.{self.region}.run.app",
                    "aster-enhanced-dashboard": f"https://aster-enhanced-dashboard-{self.project_num}.{self.region}.run.app"
                }

            def check_service_status(self, service_name: str):
                """Check status of a service"""
                try:
                    import urllib.request
                    import json
                    url = f"{self.services[service_name]}/status"
                    with urllib.request.urlopen(url, timeout=10) as response:
                        if response.status == 200:
                            data = response.read().decode('utf-8')
                            return json.loads(data)
                        else:
                            return {"status": "error", "error": f"HTTP {response.status}"}
                except Exception as e:
                    print(f"Error checking {service_name} status: {e}")
                    return {"status": "error", "error": str(e)}

            def start_service(self, service_name: str):
                """Start a trading service"""
                try:
                    import urllib.request
                    import json
                    url = f"{self.services[service_name]}/start"
                    req = urllib.request.Request(url, method='POST')
                    with urllib.request.urlopen(req, timeout=30) as response:
                        if response.status == 200:
                            data = response.read().decode('utf-8')
                            result = json.loads(data) if data else {}
                            return {"status": "success", "message": f"{service_name} started", **result}
                        else:
                            return {"status": "error", "error": f"HTTP {response.status}"}
                except Exception as e:
                    print(f"Error starting {service_name}: {e}")
                    return {"status": "error", "error": str(e)}

            def stop_service(self, service_name: str):
                """Stop a trading service"""
                try:
                    import urllib.request
                    import json
                    url = f"{self.services[service_name]}/stop"
                    req = urllib.request.Request(url, method='POST')
                    with urllib.request.urlopen(req, timeout=30) as response:
                        if response.status == 200:
                            data = response.read().decode('utf-8')
                            result = json.loads(data) if data else {}
                            return {"status": "success", "message": f"{service_name} stopped", **result}
                        else:
                            return {"status": "error", "error": f"HTTP {response.status}"}
                except Exception as e:
                    print(f"Error stopping {service_name}: {e}")
                    return {"status": "error", "error": str(e)}

            def get_service_performance(self, service_name: str):
                """Get performance data from a service"""
                try:
                    import urllib.request
                    import json
                    url = f"{self.services[service_name]}/performance"
                    with urllib.request.urlopen(url, timeout=10) as response:
                        if response.status == 200:
                            data = response.read().decode('utf-8')
                            return json.loads(data)
                        else:
                            return {"status": "error", "error": f"HTTP {response.status}"}
                except Exception as e:
                    print(f"Error getting {service_name} performance: {e}")
                    return {"status": "error", "error": str(e)}

            def get_service_positions(self, service_name: str):
                """Get positions data from a service"""
                try:
                    import urllib.request
                    import json
                    url = f"{self.services[service_name]}/positions"
                    with urllib.request.urlopen(url, timeout=10) as response:
                        if response.status == 200:
                            data = response.read().decode('utf-8')
                            return json.loads(data)
                        else:
                            return {"status": "error", "error": f"HTTP {response.status}"}
                except Exception as e:
                    print(f"Error getting {service_name} positions: {e}")
                    return {"status": "error", "error": str(e)}

            def get_combined_status(self):
                """Get combined status from all trading services"""
                try:
                    # Simple status check - just try to connect to the health endpoints
                    import urllib.request
                    import json

                    trading_status = {"status": "unknown"}
                    self_learning_status = {"status": "unknown"}

                    # Check trading agent
                    try:
                        with urllib.request.urlopen(f"{self.services['aster-trading-agent']}/status", timeout=5) as response:
                            if response.status == 200:
                                data = json.loads(response.read().decode('utf-8'))
                                trading_status = {"status": "running" if data.get("status") == "running" else "stopped"}
                    except Exception as e:
                        trading_status = {"status": "error", "error": str(e)}

                    # Check self-learning trader (currently having issues)
                    self_learning_status = {"status": "disabled", "message": "Service temporarily disabled due to container issues"}

                    # Determine overall status
                    if trading_status.get("status") == "running":
                        overall_status = "running"
                    else:
                        overall_status = "stopped"

                except Exception as e:
                    print(f"Error in get_combined_status: {e}")
                    overall_status = "error"
                    trading_status = {"status": "error", "error": str(e)}
                    self_learning_status = {"status": "error", "error": str(e)}

                return {
                    "service": "Self-Learning Trading Bot",
                    "version": "2.0.0",
                    "status": overall_status,
                    "features": [
                        "Self-learning ML models",
                        "Aggressive perpetual trading",
                        "Multiple trading strategies",
                        "Real-time adaptation",
                        "Risk management"
                    ],
                    "endpoints": {
                        "health": "/health",
                        "status": "/status",
                        "performance": "/performance",
                        "positions": "/positions",
                        "market_data": "/market-data",
                        "strategy_weights": "/strategy-weights",
                        "learning_status": "/learning-status",
                        "start": "POST /start",
                        "stop": "POST /stop",
                        "manual_trade": "POST /manual-trade"
                    },
                    "services": {
                        "aster_trading_agent": trading_status,
                        "aster_self_learning_trader": self_learning_status
                    }
                }

            def start_all_trading(self):
                """Start all trading services"""
                results = {}

                # Start main trading agent
                print("Starting aster-trading-agent...")
                results["aster_trading_agent"] = self.start_service("aster-trading-agent")

                # Start self-learning trader
                print("Starting aster-self-learning-trader...")
                results["aster_self_learning_trader"] = self.start_service("aster-self-learning-trader")

                return {
                    "status": "completed",
                    "results": results,
                    "message": "Trading services start initiated"
                }

            def stop_all_trading(self):
                """Stop all trading services"""
                results = {}

                # Stop main trading agent
                print("Stopping aster-trading-agent...")
                results["aster_trading_agent"] = self.stop_service("aster-trading-agent")

                # Stop self-learning trader
                print("Stopping aster-self-learning-trader...")
                results["aster_self_learning_trader"] = self.stop_service("aster-self-learning-trader")

                return {
                    "status": "completed",
                    "results": results,
                    "message": "Trading services stop initiated"
                }

            def get_combined_performance(self):
                """Get combined performance from all services"""
                trading_perf = self.get_service_performance("aster-trading-agent")
                self_learning_perf = self.get_service_performance("aster-self-learning-trader")

                # Combine performance data
                combined = {
                    "trading_agent": trading_perf,
                    "self_learning_trader": self_learning_perf,
                    "combined_metrics": {}
                }

                # Calculate combined metrics if both have data
                if (trading_perf.get("status") != "error" and
                    self_learning_perf.get("status") != "error"):

                    # Simple combination - you can make this more sophisticated
                    combined["combined_metrics"] = {
                        "total_pnl": (trading_perf.get("total_pnl", 0) +
                                    self_learning_perf.get("total_pnl", 0)),
                        "total_trades": (trading_perf.get("total_trades", 0) +
                                       self_learning_perf.get("total_trades", 0))
                    }

                return combined

        cloud_controller = CloudServiceController()
        logger.info("Cloud service controller initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize cloud controller: {e}")
        return False

# Initialize cloud controller on startup
initialize_cloud_controller()

class TradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    strategy: str

class PerformanceMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    balance: float
    active_positions: int

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "self-learning-trading-bot",
        "running": is_running,
        "version": "2.0.0",
        "cloud_controller": cloud_controller is not None
    }

@app.get("/test-cloud")
async def test_cloud():
    """Test cloud controller functionality"""
    if not cloud_controller:
        return {"error": "Cloud controller not initialized"}

    try:
        # Test checking status
        status = cloud_controller.check_service_status("aster-trading-agent")
        return {
            "cloud_controller": "initialized",
            "test_status": status,
            "trading_agent_url": cloud_controller.services.get("aster-trading-agent")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
async def get_status():
    """Get trading bot status"""
    if not trading_bot:
        return {"status": "not_initialized"}
    
    return {
        "status": "running" if is_running else "stopped",
        "timestamp": datetime.now().isoformat(),
        "bot_type": "self_learning_aggressive",
        "balance": trading_bot.balance,
        "active_positions": len(trading_bot.positions),
        "total_trades": trading_bot.performance_metrics['total_trades']
    }

@app.get("/performance")
async def get_performance():
    """Get detailed performance metrics"""
    if cloud_controller:
        try:
            perf_data = cloud_controller.get_combined_performance()
            return perf_data
        except Exception as e:
            logger.error(f"Failed to get cloud performance: {e}")

    # Fallback to local implementation
    if not trading_bot:
        raise HTTPException(status_code=404, detail="Trading bot not initialized")

    metrics = trading_bot.performance_metrics
    return PerformanceMetrics(
        total_trades=metrics['total_trades'],
        winning_trades=metrics['winning_trades'],
        losing_trades=metrics['losing_trades'],
        total_pnl=metrics['total_pnl'],
        win_rate=metrics['win_rate'],
        balance=trading_bot.balance,
        active_positions=len(trading_bot.positions)
    )

@app.get("/positions")
async def get_positions():
    """Get current positions"""
    if cloud_controller:
        try:
            # Get positions from both services
            trading_positions = cloud_controller.get_service_positions("aster-trading-agent")
            self_learning_positions = cloud_controller.get_service_positions("aster-self-learning-trader")

            combined = {
                "trading_agent": trading_positions,
                "self_learning_trader": self_learning_positions
            }
            return combined
        except Exception as e:
            logger.error(f"Failed to get cloud positions: {e}")

    # Fallback to local implementation
    if not trading_bot:
        return []
    
    positions = []
    for trade_id, trade in trading_bot.positions.items():
        positions.append({
            "id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "leverage": trade.leverage,
            "confidence": trade.confidence,
            "strategy": trade.strategy,
            "timestamp": trade.timestamp.isoformat(),
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit
        })
    
    return positions

@app.get("/market-data")
async def get_market_data():
    """Get current market data"""
    if not trading_bot:
        return {}
    
    return trading_bot.market_data

@app.get("/strategy-weights")
async def get_strategy_weights():
    """Get current strategy weights"""
    if not trading_bot:
        return {}

    return trading_bot.strategy_weights

@app.get("/test-api")
async def test_api_connectivity():
    """Test API connectivity and return detailed diagnostics."""
    try:
        from mcp_trader.execution.aster_client import AsterClient

        # Get API keys
        api_key = os.getenv("ASTER_API_KEY")
        secret_key = os.getenv("ASTER_API_SECRET")

        if not api_key or not secret_key:
            return {
                "status": "error",
                "message": "API keys not configured",
                "api_key_set": api_key is not None,
                "secret_key_set": secret_key is not None
            }

        # Test basic connectivity
        try:
            aster_client = AsterClient(api_key=api_key, secret_key=secret_key)

            # Test server time (lightweight endpoint)
            server_time = await aster_client.get_server_time()
            logger.info(f"Server time: {server_time}")

            # Test account info
            try:
                account_info = await aster_client.get_account_info()
                logger.info(f"Account balance: {account_info.total_balance}")

                return {
                    "status": "success",
                    "message": "API connectivity verified",
                    "server_time": server_time,
                    "account_balance": account_info.total_balance,
                    "available_balance": account_info.available_balance,
                    "total_positions": len(account_info.positions)
                }
            except Exception as e:
                logger.error(f"Account info error: {e}")
                return {
                    "status": "partial",
                    "message": "Server connectivity OK, but account access failed",
                    "server_time": server_time,
                    "error": str(e)
                }

        except Exception as e:
            logger.error(f"API connection error: {e}")
            return {
                "status": "error",
                "message": f"Failed to connect to Aster DEX: {str(e)}",
                "error": str(e)
            }

    except ImportError as e:
        return {
            "status": "error",
            "message": f"Import error: {str(e)}",
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "error": str(e)
        }

@app.post("/start")
async def start_trading():
    """Start the trading services"""
    if cloud_controller:
        try:
            result = cloud_controller.start_all_trading()
            logger.info("Trading services start initiated via cloud controller")
            return {
                "status": "started",
                "message": "Cloud trading services start initiated",
                "details": result
            }
        except Exception as e:
            logger.error(f"Failed to start via cloud controller: {e}")
            return {"status": "error", "message": f"Cloud controller error: {str(e)}"}

    # Fallback to local implementation
    global is_running, trading_bot

    if is_running:
        return {"status": "already_running"}
    
    try:
        # Get API keys
        api_key = os.getenv("ASTER_API_KEY")
        secret_key = os.getenv("ASTER_API_SECRET")
        
        if not api_key or not secret_key:
            logger.warning("API keys not found, using demo mode")
            api_key = "demo_key"
            secret_key = "demo_secret"
        
        if _self_learning_available:
            # Create self-learning trading configuration
            config = TradingConfig(
                initial_capital=100.0,
                max_position_size=0.3,  # 30% per trade
                max_leverage=10.0,      # Aggressive leverage
                stop_loss_pct=0.015,    # 1.5% stop loss
                take_profit_pct=0.03,   # 3% take profit
                trading_pairs=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT'],
                min_confidence=0.7,
                max_trades_per_hour=20,
                cooldown_seconds=30
            )
            
            # Initialize self-learning trading bot
            trading_bot = SelfLearningTrader(config, api_key, secret_key)
            
            # Start trading in background
            asyncio.create_task(trading_bot.start_trading())
            is_running = True
            
            logger.info("Self-learning trading bot started successfully")
            return {
                "status": "started", 
                "message": "Self-learning aggressive trading bot is now running",
                "config": {
                    "initial_capital": config.initial_capital,
                    "max_leverage": config.max_leverage,
                    "trading_pairs": config.trading_pairs,
                    "strategies": list(trading_bot.strategy_weights.keys())
                }
            }
        else:
            # Fallback to basic trading agent
            config = BasicTradingConfig(
                initial_capital=100.0,
                position_size_pct=0.1,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                daily_loss_limit_pct=0.10,
                trading_pairs=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT']
            )
            
            from mcp_trader.execution.aster_client import AsterClient
            aster_client = AsterClient(api_key=api_key, secret_key=secret_key)
            trading_bot = LiveTradingAgent(config, aster_client)
            
            # Start trading in background
            asyncio.create_task(trading_bot.start_trading())
            is_running = True
            
            logger.info("Fallback trading agent started successfully")
            return {
                "status": "started", 
                "message": "Fallback trading agent is now running",
                "config": {
                    "initial_capital": config.initial_capital,
                    "trading_pairs": config.trading_pairs
                }
            }
        
    except Exception as e:
        logger.error(f"Failed to start trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start trading bot: {str(e)}")

@app.post("/stop")
async def stop_trading():
    """Stop the trading services"""
    if cloud_controller:
        try:
            result = cloud_controller.stop_all_trading()
            logger.info("Trading services stop initiated via cloud controller")
            return {
                "status": "stopped",
                "message": "Cloud trading services stop initiated",
                "details": result
            }
        except Exception as e:
            logger.error(f"Failed to stop via cloud controller: {e}")
            return {"status": "error", "message": f"Cloud controller error: {str(e)}"}

    # Fallback to local implementation
    global is_running, trading_bot

    if not is_running:
        return {"status": "not_running"}
    
    try:
        if trading_bot:
            await trading_bot.stop_trading()
        is_running = False
        
        logger.info("Trading bot stopped")
        return {"status": "stopped", "message": "Trading bot has been stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop trading bot: {str(e)}")

@app.post("/manual-trade")
async def execute_manual_trade(trade_request: TradeRequest):
    """Execute a manual trade"""
    if not trading_bot or not is_running:
        raise HTTPException(status_code=400, detail="Trading bot not running")
    
    try:
        # Get current price
        if trade_request.symbol not in trading_bot.market_data:
            raise HTTPException(status_code=400, detail=f"No market data for {trade_request.symbol}")
        
        current_price = trading_bot.market_data[trade_request.symbol]['price']
        
        # Execute trade
        trade = await trading_bot.execute_trade(
            symbol=trade_request.symbol,
            side=trade_request.side,
            quantity=trade_request.quantity,
            price=current_price,
            strategy=trade_request.strategy,
            confidence=0.8  # Manual trade confidence
        )
        
        if trade:
            return {
                "status": "success",
                "trade_id": trade.id,
                "message": f"Trade executed: {trade_request.side} {trade_request.quantity} {trade_request.symbol}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to execute trade")
            
    except Exception as e:
        logger.error(f"Error executing manual trade: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing trade: {str(e)}")

@app.get("/learning-status")
async def get_learning_status():
    """Get learning and model status"""
    if not trading_bot:
        return {"status": "not_initialized"}
    
    return {
        "memory_buffer_size": len(trading_bot.memory_buffer),
        "last_model_update": trading_bot.last_model_update,
        "strategy_weights": trading_bot.strategy_weights,
        "models_loaded": len([m for m in trading_bot.models.values() if hasattr(m, 'coef_')]),
        "learning_active": len(trading_bot.memory_buffer) >= trading_bot.config.batch_size
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    print(f"Cloud controller initialized: {cloud_controller is not None}")

    if cloud_controller:
        # Get real status from cloud services
        try:
            print("Getting combined status from cloud services...")
            status_info = cloud_controller.get_combined_status()
            print(f"Status info: {status_info}")
            return status_info
        except Exception as e:
            print(f"Failed to get cloud status: {e}")
            logger.error(f"Failed to get cloud status: {e}")
            # Fallback to local status

    # Fallback response
    print("Using fallback response")
    return {
        "service": "Self-Learning Trading Bot",
        "version": "2.0.0",
        "status": "running" if is_running else "stopped",
        "features": [
            "Self-learning ML models",
            "Aggressive perpetual trading",
            "Multiple trading strategies",
            "Real-time adaptation",
            "Risk management"
        ],
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "performance": "/performance",
            "positions": "/positions",
            "market_data": "/market-data",
            "strategy_weights": "/strategy-weights",
            "learning_status": "/learning-status",
            "start": "POST /start",
            "stop": "POST /stop",
            "manual_trade": "POST /manual-trade"
        }
    }

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if trading_bot:
        asyncio.create_task(trading_bot.stop_trading())
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting Self-Learning Trading Bot server on port {port}")
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
