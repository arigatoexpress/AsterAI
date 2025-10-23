#!/usr/bin/env python3
"""
Economical Cloud Run Trading Bot
Aggressive strategies with 10% max position, 5 max open positions
Perfect for testing with minimal costs
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Import our trading components
from mcp_trader.execution.aster_client import AsterClient, AsterConfig
from mcp_trader.risk.risk_manager import RiskManager
from strategies.aggressive_perps_strategy import AggressivePerpsStrategy
from mcp_trader.trading.types import PortfolioState, MarketRegime
from config_aggressive_trading import get_aggressive_config, apply_environment_overrides

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
app = FastAPI(title="AsterAI Economical HFT Bot")
trading_bot = None
shutdown_event = asyncio.Event()

class EconomicalTradingBot:
    """Lightweight trading bot for economical Cloud Run deployment."""

    def __init__(self):
        # Load aggressive configuration
        self.config = apply_environment_overrides(get_aggressive_config())
        self.max_position_size = self.config["max_position_size"]
        self.max_open_positions = self.config["max_open_positions"]
        self.initial_capital = 1000.0  # Start small for testing

            # Initialize components
        self.aster_client = None
        self.risk_manager = None
        self.aggressive_strategy = AggressivePerpsStrategy(
            max_positions=self.config["max_open_positions"],
            max_position_size=self.config["max_position_size"]
        )

        # Enable live trading mode (remove paper trading safety)
        self.paper_trading = os.getenv("ENABLE_PAPER_TRADING", "false").lower() == "true"

        # Portfolio state
        self.portfolio = PortfolioState(
            total_balance=self.initial_capital,
            total_positions_value=0.0,
            positions={}
        )

        # Trading statistics
        self.stats = {
            "trades_executed": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "active_positions": 0,
            "start_time": datetime.now(),
            "last_update": datetime.now()
        }

    async def initialize(self):
        """Initialize the trading bot."""
        try:
            logger.info("🚀 Initializing Economical Trading Bot...")

            # Initialize Aster API client
            api_key = os.getenv("ASTER_API_KEY")
            secret_key = os.getenv("ASTER_SECRET_KEY")

            if not api_key or not secret_key:
                logger.warning("⚠️  Aster API keys not found - running in simulation mode")
                self.paper_trading = True
            else:
                config = AsterConfig(
                    api_key=api_key,
                    secret_key=secret_key
                )
                self.aster_client = AsterClient(config)
                self.paper_trading = os.getenv("ENABLE_PAPER_TRADING", "true").lower() == "true"

            # Initialize risk manager with aggressive settings
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.config["max_portfolio_risk"],
                max_single_position_risk=self.config["max_single_position_risk"]
            )

            # Aggressive perpetual strategy is already initialized

            logger.info("✅ Trading bot initialized successfully")
            logger.info(f"🎯 Risk Parameters: {self.max_position_size*100}% max position, {self.max_open_positions} max positions")
            logger.info(f"📊 Mode: {'Paper Trading' if self.paper_trading else 'Live Trading'}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize trading bot: {e}")
            raise

    async def execute_trading_cycle(self):
        """Execute one trading cycle."""
        try:
            # Get market data (simplified for economical deployment)
            market_data = await self.get_market_data()

            if not market_data:
                logger.warning("No market data available")
                return

            # Update portfolio state
            self.portfolio = await self.update_portfolio_state()

            # Check risk limits
            risk_metrics = self.risk_manager.assess_portfolio_risk(self.portfolio)

            # Execute aggressive perpetual strategy
            try:
                # Update market data in strategy
                for symbol, data in market_data.items():
                    self.aggressive_strategy.update_market_data(symbol, data)

                # Check for entry signals
                for symbol in market_data.keys():
                    if len(self.aggressive_strategy.positions) >= self.max_open_positions:
                        logger.info("⚠️  Max positions reached, skipping new trades")
                        break

                    entry_signal = self.aggressive_strategy.should_enter_position(symbol, market_data[symbol])
                    if entry_signal:
                        success = self.aggressive_strategy.execute_entry(entry_signal)
                        if success:
                            self.stats["trades_executed"] += 1
                            logger.info(f"🚀 Aggressive entry executed for {symbol}")

                # Check for exit signals
                positions_to_exit = []
                for symbol, position in self.aggressive_strategy.positions.items():
                    current_price = market_data.get(symbol, {}).get('prices', [position.entry_price])[-1]
                    exit_reason = self.aggressive_strategy.should_exit_position(position, current_price, market_data.get(symbol, {}))
                    if exit_reason:
                        success = self.aggressive_strategy.execute_exit(symbol, exit_reason)
                        if success:
                            positions_to_exit.append(symbol)

                # Update portfolio after exits
                for symbol in positions_to_exit:
                    if symbol in self.portfolio.positions:
                        del self.portfolio.positions[symbol]

            except Exception as e:
                logger.error(f"Aggressive strategy error: {e}")

            # Update statistics
            self.update_statistics()

        except Exception as e:
            logger.error(f"❌ Trading cycle error: {e}")

    async def get_market_data(self) -> Dict[str, Any]:
        """Get market data focused on mid/small caps for maximum profit potential."""
        # PRIORITY: Mid/Small caps for explosive gains
        symbols = [
            "SOLUSDT", "SUIUSDT", "ASTERUSDT", "PENGUUSDT",  # Mid caps
            "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT"   # Small caps
        ]

        market_data = {}
        for symbol in symbols:
            try:
                if self.aster_client and not self.paper_trading:
                    # Get real data
                    ticker = await self.aster_client.get_24hr_ticker(symbol)
                    market_data[symbol] = {
                        "price": float(ticker.get("lastPrice", 0)),
                        "volume": float(ticker.get("volume", 0)),
                        "change_24h": float(ticker.get("priceChangePercent", 0))
                    }
                else:
                    # Generate realistic simulated data
                    import random
                    base_prices = {"BTCUSDT": 50000, "ETHUSDT": 3000, "SOLUSDT": 100}
                    base_price = base_prices.get(symbol, 1000)
                    price_change = random.uniform(-0.02, 0.02)  # ±2%
                    market_data[symbol] = {
                        "price": base_price * (1 + price_change),
                        "volume": random.uniform(1000, 10000),
                        "change_24h": price_change * 100
                    }

            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
                continue

        return market_data

    async def update_portfolio_state(self) -> PortfolioState:
        """Update portfolio state."""
        if self.aster_client and not self.paper_trading:
            try:
                # Get real account info
                account_info = await self.aster_client.get_account_info()
                # Update portfolio from real data
                pass
            except Exception as e:
                logger.error(f"Error updating portfolio: {e}")

        # For now, return current portfolio (enhance with real data later)
        return self.portfolio

    async def execute_signal(self, signal: Dict[str, Any], risk_metrics):
        """Execute a trading signal."""
        try:
            symbol = signal["symbol"]
            side = signal["side"]
            quantity = signal["quantity"]

            # Check risk limits
            if risk_metrics.portfolio_value * self.max_position_size < quantity * signal.get("price", 0):
                logger.info(f"⚠️  Position size too large for {symbol}")
                return

            if self.paper_trading:
                # Simulate trade execution
                price = signal.get("price", 1000)
                pnl = (price * 0.001) * (1 if side == "buy" else -1)  # Simulate small P&L

                self.portfolio.positions[symbol] = {
                    "quantity": quantity,
                    "entry_price": price,
                    "current_price": price,
                    "pnl": pnl
                }

                self.stats["trades_executed"] += 1
                self.stats["total_pnl"] += pnl

                logger.info(f"📈 Executed {side} order for {symbol}: ${quantity * price:.4f}")
            else:
                # Execute real trade
                if self.aster_client:
                    # Implement real trade execution
                    pass

        except Exception as e:
            logger.error(f"Error executing signal: {e}")

    def update_statistics(self):
        """Update trading statistics."""
        self.stats["active_positions"] = len(self.portfolio.positions)
        self.stats["last_update"] = datetime.now()

        total_trades = self.stats["trades_executed"]
        if total_trades > 0:
            # Simple win rate calculation (mock for now)
            self.stats["win_rate"] = 0.55  # Assume 55% win rate

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring."""
        portfolio_status = self.aggressive_strategy.get_portfolio_status()

        return {
            "status": "healthy",
            "mode": "paper_trading" if self.paper_trading else "live_trading",
            "strategy": "aggressive_perpetuals",
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "active_positions": portfolio_status["active_positions"],
            "max_positions": portfolio_status["max_positions"],
            "total_exposure": portfolio_status["total_exposure"],
            "total_pnl": self.stats["total_pnl"],
            "trades_executed": self.stats["trades_executed"],
            "win_rate": self.stats["win_rate"],
            "max_position_size": self.max_position_size,
            "max_open_positions": self.max_open_positions,
            "daily_profit_target": self.config["daily_profit_target"],
            "max_daily_loss": self.config["max_daily_loss"],
            "target_symbols": len(self.config["target_symbols"]),
            "positions": portfolio_status["positions"],
            "last_update": self.stats["last_update"].isoformat()
        }

# Global bot instance
trading_bot = EconomicalTradingBot()

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    await trading_bot.initialize()

    # Start trading loop in background
    asyncio.create_task(trading_loop())

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        status = await trading_bot.get_health_status()
        return JSONResponse(content=status, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500
        )

@app.get("/stats")
async def get_stats():
    """Get trading statistics."""
    return await trading_bot.get_health_status()

@app.get("/signals/{symbol}")
async def get_signals(symbol: str):
    """Get trading signals for a specific symbol."""
    try:
        signal_summary = trading_bot.aggressive_strategy.get_signal_summary(symbol.upper())
        return JSONResponse(content=signal_summary, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to get signals for {symbol}: {str(e)}"},
            status_code=500
        )

@app.post("/force-entry/{symbol}")
async def force_entry(symbol: str):
    """Force entry for testing (paper trading only)."""
    if not trading_bot.paper_trading:
        return JSONResponse(
            content={"error": "Force entry only available in paper trading mode"},
            status_code=403
        )

    try:
        # Create mock market data for testing
        mock_data = {
            'prices': [50000, 50100, 50200, 50300, 50400, 50500],
            'volumes': [1000, 1200, 1500, 1800, 2000, 2500],
            'funding_rate': 0.0002
        }

        entry_signal = trading_bot.aggressive_strategy.should_enter_position(symbol.upper(), mock_data)
        if entry_signal:
            success = trading_bot.aggressive_strategy.execute_entry(entry_signal)
            return JSONResponse(
                content={
                    "success": success,
                    "signal": entry_signal,
                    "message": f"Force entry executed for {symbol}" if success else "Entry failed"
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={"success": False, "message": "No entry signal generated"},
                status_code=200
            )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Force entry failed: {str(e)}"},
            status_code=500
        )

@app.get("/dashboard")
async def dashboard():
    """Aggressive perpetual trading dashboard."""
    stats = await trading_bot.get_health_status()

    # Build positions table
    positions_html = ""
    if stats['positions']:
        positions_html = "<h3>Active Positions</h3><table border='1'><tr><th>Symbol</th><th>Side</th><th>Leverage</th><th>Entry Price</th><th>Quantity</th><th>Hours Held</th></tr>"
        for pos in stats['positions']:
            positions_html += f"<tr><td>{pos['symbol']}</td><td>{pos['side']}</td><td>{pos['leverage']}x</td><td>{pos['entry_price']:.2f}</td><td>{pos['quantity']:.4f}</td><td>{pos['hours_held']:.1f}</td></tr>"
        positions_html += "</table>"
    else:
        positions_html = "<p>No active positions</p>"

    html = f"""
    <html>
    <head><title>AsterAI Aggressive Perps Bot</title></head>
    <body>
        <h1>🚀 AsterAI Aggressive Perpetual Trading Bot</h1>
        <h2>Strategy: Maximum Profit Potential</h2>
        <h3>Status: {stats['status']} | Mode: {stats['mode']}</h3>

        <div style="display: flex; gap: 20px;">
        <div>
            <h3>📊 Performance</h3>
            <p>Trades Executed: {stats['trades_executed']}</p>
            <p>Total P&L: ${stats['total_pnl']:.2f}</p>
            <p>Active Positions: {stats['active_positions']}/{stats['max_positions']}</p>
            <p>Total Exposure: {stats['total_exposure']:.2f}x</p>
            <p>Win Rate: {stats['win_rate']:.1%}</p>
            <p>Uptime: {stats['uptime_seconds']:.0f} seconds</p>
        </div>

        <div>
            <h3>🎯 Strategy Parameters</h3>
            <p>Max Position Size: {trading_bot.max_position_size:.0%}</p>
            <p>Max Open Positions: {trading_bot.max_open_positions}</p>
            <p>Daily Profit Target: {stats['daily_profit_target']:.0%}</p>
            <p>Max Daily Loss: {stats['max_daily_loss']:.0%}</p>
            <p>Target Symbols: {stats['target_symbols']}</p>
        </div>
        </div>

        {positions_html}

        <h3>🔗 API Endpoints</h3>
        <ul>
            <li><a href="/health">/health</a> - Health check</li>
            <li><a href="/stats">/stats</a> - Detailed stats (JSON)</li>
            <li><a href="/signals/BTCUSDT">/signals/BTCUSDT</a> - Signal analysis</li>
        </ul>

        <p><small>Last Update: {stats['last_update']}</small></p>
        <p><small>💰 Cost: $5-15/month | 🎯 Focus: Aggressive calculated bets</small></p>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)

async def trading_loop():
    """Main trading loop."""
    logger.info("🎯 Starting economical trading loop...")

    while not shutdown_event.is_set():
        try:
            await trading_bot.execute_trading_cycle()
            await asyncio.sleep(60)  # Execute every minute (economical)

        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await asyncio.sleep(10)  # Shorter delay on error

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("🛑 Shutdown signal received")
    shutdown_event.set()

if __name__ == "__main__":
    # Handle shutdown signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get port from environment
    port = int(os.getenv("PORT", 8080))

    # Run the FastAPI app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
