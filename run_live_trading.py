#!/usr/bin/env python3
"""
🚀 Aster AI Live Trading Bot Runner
Generated: 2025-10-20

Simple runner script for the live trading bot with proper configuration.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_trading_agent import LiveTradingAgent, TradingConfig
from mcp_trader.execution.aster_client import AsterClient

def setup_logging():
    """Set up logging configuration."""
    log_filename = f"logs/live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("🚀 Aster AI Live Trading Bot Started")
    logger.info(f"📝 Log file: {log_filename}")
    return logger

def load_api_credentials():
    """Load API credentials from config file."""
    try:
        import json
        with open('.api_keys.json', 'r') as f:
            keys = json.load(f)
        return keys.get('aster_api_key'), keys.get('aster_api_secret')
    except Exception as e:
        print(f"⚠️  Warning: Could not load API credentials: {e}")
        print("   Running in simulation mode only")
        return None, None

async def main():
    """Main function to run the live trading bot."""

    print("="*80)
    print("🤖 ASTER AI LIVE TRADING BOT")
    print("="*80)

    # Setup logging
    logger = setup_logging()

    # Load configuration
    config = TradingConfig(
        initial_capital=100.0,  # $100 starting capital
        max_leverage=3.0,
        position_size_pct=0.02,  # 2% per trade
        stop_loss_pct=0.02,      # 2% stop loss
        take_profit_pct=0.04,    # 4% take profit
        daily_loss_limit_pct=0.10,  # 10% daily loss limit
        max_positions=2,         # Max 2 positions at once
        trading_pairs=["BTCUSDT", "ETHUSDT"],
        dry_run=True  # Start in dry-run mode for safety
    )

    print("⚙️  Configuration:")
    print(f"   • Capital: ${config.initial_capital}")
    print(f"   • Max Positions: {config.max_positions}")
    print(f"   • Position Size: {config.position_size_pct*100}%")
    print(f"   • Stop Loss: {config.stop_loss_pct*100}%")
    print(f"   • Take Profit: {config.take_profit_pct*100}%")
    print(f"   • Trading Pairs: {', '.join(config.trading_pairs)}")
    print(f"   • Dry Run: {config.dry_run}")

    # Load API credentials
    api_key, api_secret = load_api_credentials()

    if api_key and api_secret:
        # Create real Aster client
        aster_client = AsterClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url="https://api.aster.exchange"
        )
        config.dry_run = False  # Disable dry run if we have real credentials
        print("✅ Real trading mode enabled")
    else:
        # Create mock client for dry-run mode
        class MockAsterClient:
            def __init__(self):
                self.positions = {}
                self.balance = config.initial_capital

            async def get_24hr_ticker(self, symbol):
                # Return mock ticker data
                return {
                    'lastPrice': '50000.0' if 'BTC' in symbol else '3000.0',
                    'symbol': symbol
                }

            async def get_account_info(self):
                return {'total_balance': str(self.balance)}

        aster_client = MockAsterClient()
        config.dry_run = True
        print("🎭 Simulation mode enabled (no real API credentials)")

    print("\n🚀 Initializing Live Trading Agent...")

    # Create and start trading agent
    try:
        agent = LiveTradingAgent(config, aster_client)

        print("✅ Trading agent initialized successfully")
        print("\n🛡️  Safety Features Active:")
        print("   • Emergency stop capability")
        print("   • Daily loss limits")
        print("   • Position size controls")
        print("   • Stop-loss protection")
        print("   • Risk management")

        print("\n🎯 Starting live trading...")
        print("   Press Ctrl+C to stop gracefully")

        # Start trading loop
        await agent.start_trading()

    except KeyboardInterrupt:
        print("\n⏹️  Trading stopped by user")
        if 'agent' in locals():
            await agent.stop_trading()

    except Exception as e:
        logger.error(f"❌ Trading failed: {e}")
        if 'agent' in locals():
            await agent.emergency_stop_trading()
        raise

    finally:
        print("\n💾 Saving final trading session...")
        if 'agent' in locals():
            status = agent.get_status()
            print("📊 Final Status:")
            print(f"   • Total P&L: ${status['metrics']['total_pnl']:.2f}")
            print(f"   • Total Trades: {status['metrics']['total_trades']}")
            print(f"   • Win Rate: {status['metrics']['win_rate']:.1%}")
            print(f"   • Active Positions: {status['positions']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
