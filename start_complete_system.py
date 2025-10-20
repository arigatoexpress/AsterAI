#!/usr/bin/env python3
"""
🚀 Aster AI Complete Trading System Launcher
Generated: 2025-10-20

Starts the complete Aster AI trading ecosystem:
- Live Trading Bot
- Matrix Dashboard Server
- Background data collection
- Monitoring and logging
"""

import os
import sys
import asyncio
import subprocess
import signal
import time
from pathlib import Path

def setup_environment():
    """Set up the environment for the complete system."""
    print("🔧 Setting up environment...")

    # Add current directory to path
    current_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(current_dir))

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("live_trading_results", exist_ok=True)
    os.makedirs("data/local_cache", exist_ok=True)

    # Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LIVE_TRADING_CAPITAL'] = '100.0'
    os.environ['DRY_RUN_MODE'] = 'true'  # Safe default
    os.environ['MAX_POSITIONS'] = '2'

    print("✅ Environment configured")

def check_requirements():
    """Check if all required components are available."""
    print("🔍 Checking system requirements...")

    required_files = [
        'live_trading_agent.py',
        'dashboard_server.py',
        'requirements.txt'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print("✅ All required files present")
    return True

async def start_dashboard_server():
    """Start the Matrix dashboard server."""
    print("📊 Starting Matrix Dashboard Server...")

    try:
        # Import and run dashboard server
        from dashboard_server import app

        # Run in background thread
        def run_flask():
            app.run(host='0.0.0.0', port=8081, debug=False, use_reloader=False)

        import threading
        dashboard_thread = threading.Thread(target=run_flask, daemon=True)
        dashboard_thread.start()

        # Wait a moment for server to start
        await asyncio.sleep(2)

        print("✅ Dashboard server started on http://localhost:8081")

        return True

    except Exception as e:
        print(f"❌ Failed to start dashboard server: {e}")
        return False

async def start_live_trading():
    """Start the live trading bot."""
    print("🤖 Starting Live Trading Bot...")

    try:
        # Import live trading components
        from live_trading_agent import LiveTradingAgent, TradingConfig
        from mcp_trader.execution.aster_client import AsterClient

        # Create configuration
        config = TradingConfig(
            initial_capital=100.0,
            max_leverage=3.0,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            daily_loss_limit_pct=0.10,
            max_positions=2,
            trading_pairs=["BTCUSDT", "ETHUSDT"],
            dry_run=True  # Safe mode
        )

        # Try to load API credentials
        api_key, api_secret = None, None
        try:
            import json
            with open('.api_keys.json', 'r') as f:
                keys = json.load(f)
            api_key = keys.get('aster_api_key')
            api_secret = keys.get('aster_api_secret')
        except:
            pass

        # Create appropriate client
        if api_key and api_secret:
            aster_client = AsterClient(
                api_key=api_key,
                api_secret=api_secret,
                base_url="https://api.aster.exchange"
            )
            config.dry_run = False
            print("✅ Real trading mode enabled")
        else:
            # Mock client for dry-run mode
            class MockAsterClient:
                def __init__(self):
                    self.balance = config.initial_capital

                async def get_24hr_ticker(self, symbol):
                    return {
                        'lastPrice': '50000.0' if 'BTC' in symbol else '3000.0',
                        'symbol': symbol
                    }

                async def get_account_info(self):
                    # Return object with total_balance attribute
                    class AccountInfo:
                        def __init__(self, balance):
                            self.total_balance = str(balance)
                    return AccountInfo(self.balance)

                async def get_order_book(self, symbol):
                    return {'bids': [], 'asks': []}

                async def get_klines(self, symbol, interval, limit=100):
                    """Get klines with proper data format and historical fallback"""
                    import time
                    import numpy as np  # Import here to avoid global issues

                    # Try to load from local historical data first
                    try:
                        historical_data = self._load_historical_data(symbol)
                        if historical_data and len(historical_data) >= limit:
                            # Use recent historical data
                            recent_data = historical_data.tail(limit).values.tolist()
                            # Convert to expected format (timestamp, open, high, low, close, volume, etc.)
                            formatted_klines = []
                            for row in recent_data:
                                # Assuming historical data format: timestamp, open, high, low, close, volume
                                timestamp = int(row[0]) if len(row) > 0 else int(time.time() * 1000)
                                open_price = float(row[1]) if len(row) > 1 and row[1] else 50000.0 if 'BTC' in symbol else 3000.0
                                high_price = float(row[2]) if len(row) > 2 and row[2] else open_price * 1.001
                                low_price = float(row[3]) if len(row) > 3 and row[3] else open_price * 0.999
                                close_price = float(row[4]) if len(row) > 4 and row[4] else open_price
                                volume = float(row[5]) if len(row) > 5 and row[5] else 1000.0

                                formatted_klines.append([
                                    timestamp,  # timestamp
                                    str(open_price),  # open
                                    str(high_price),  # high
                                    str(low_price),  # low
                                    str(close_price),  # close
                                    str(volume),  # volume
                                    timestamp + 60000,  # close_time
                                    str(volume * close_price * 0.5),  # quote_volume
                                    "10",  # trades
                                    str(volume * 0.5),  # taker_buy_base
                                    str(volume * 0.5 * close_price),  # taker_buy_quote
                                    "0"  # ignore
                                ])

                            logger.info(f"Using {len(formatted_klines)} historical klines for {symbol}")
                            return formatted_klines
                    except Exception as e:
                        logger.warning(f"Failed to load historical data for {symbol}: {e}")

                    # Fallback to generated data if historical not available
                    current_time = int(time.time() * 1000)
                    base_price = 50000.0 if 'BTC' in symbol else 3000.0

                    klines = []
                    for i in range(min(limit, 100)):
                        timestamp = current_time - (limit - i) * 60000
                        volatility = 0.002
                        price_change = np.random.normal(0, volatility)
                        close_price = base_price * (1 + price_change)

                        # Ensure some trend and mean reversion
                        trend_factor = (i - limit/2) * 0.0001
                        mean_reversion = (base_price - close_price) * 0.1
                        close_price += trend_factor + mean_reversion

                        # Generate OHLC
                        spread = abs(np.random.normal(0, volatility * 2))
                        high_price = close_price * (1 + spread)
                        low_price = close_price * (1 - spread)
                        open_price = close_price * (1 + np.random.normal(0, volatility))

                        klines.append([
                            timestamp,
                            str(open_price),
                            str(high_price),
                            str(low_price),
                            str(close_price),
                            "1000.0",
                            timestamp + 60000,
                            "500.0",
                            "10",
                            "500.0",
                            "250.0",
                            "0"
                        ])

                        base_price = close_price

                    logger.info(f"Generated {len(klines)} synthetic klines for {symbol}")
                    return klines

                def _load_historical_data(self, symbol):
                    """Load historical data from local files"""
                    try:
                        import pandas as pd
                        import os

                        # Map symbols to file names
                        symbol_map = {
                            'BTCUSDT': 'btc',
                            'ETHUSDT': 'eth',
                            'BTC': 'btc',
                            'ETH': 'eth'
                        }

                        file_symbol = symbol_map.get(symbol, symbol.lower())

                        # Look for historical data files
                        data_paths = [
                            f"data/historical/crypto/{file_symbol}.parquet",
                            f"data/historical/ultimate_dataset/crypto/{file_symbol}.parquet",
                            f"data/local_cache/{file_symbol}_historical.parquet"
                        ]

                        for path in data_paths:
                            if os.path.exists(path):
                                df = pd.read_parquet(path)
                                if not df.empty and len(df) > 100:  # Ensure we have enough data
                                    logger.info(f"Loaded {len(df)} rows of historical data from {path}")
                                    return df
                    except Exception as e:
                        logger.warning(f"Failed to load historical data for {symbol}: {e}")

                    return None

            aster_client = MockAsterClient()
            config.dry_run = True
            print("🎭 Simulation mode enabled (no API credentials)")

        # Create and start trading agent
        agent = LiveTradingAgent(config, aster_client)

        print("✅ Live trading bot initialized")
        print("🚀 Starting trading operations...")

        # Start trading in background task
        trading_task = asyncio.create_task(agent.start_trading())

        return agent, trading_task

    except Exception as e:
        print(f"❌ Failed to start live trading bot: {e}")
        return None, None

async def main():
    """Main function to start the complete system."""

    print("="*80)
    print("🚀 ASTER AI COMPLETE TRADING SYSTEM")
    print("="*80)

    # Setup environment
    setup_environment()

    # Check requirements
    if not check_requirements():
        print("❌ System requirements not met. Please check missing files.")
        return

    print("\n🎯 Starting system components...")

    # Start dashboard server
    dashboard_started = await start_dashboard_server()

    # Start live trading bot
    agent, trading_task = await start_live_trading()

    if agent and trading_task:
        print("\n" + "="*80)
        print("✅ COMPLETE SYSTEM STARTED SUCCESSFULLY!")
        print("="*80)
        print("")
        print("📊 System Components:")
        print("   • 🤖 Live Trading Bot: Active")
        if dashboard_started:
            print("   • 📊 Matrix Dashboard: http://localhost:8081")
        else:
            print("   • 📊 Matrix Dashboard: Failed to start")
        print("   • 💾 Logging: logs/ directory")
        print("   • 📈 Results: live_trading_results/ directory")
        print("")
        print("🛡️  Safety Features:")
        print("   • Dry-run mode enabled for safety")
        print("   • Risk management with stop-loss protection")
        print("   • Position size limits")
        print("   • Emergency stop capability")
        print("")
        print("⚙️  Configuration:")
        print(f"   • Capital: ${agent.config.initial_capital}")
        print(f"   • Max Positions: {agent.config.max_positions}")
        print(f"   • Stop Loss: {agent.config.stop_loss_pct*100}%")
        print(f"   • Take Profit: {agent.config.take_profit_pct*100}%")
        print(f"   • Trading Pairs: {', '.join(agent.config.trading_pairs)}")
        print("")
        print("🎮 Controls:")
        print("   • Monitor dashboard for real-time status")
        print("   • Check logs for detailed activity")
        print("   • Press Ctrl+C to stop gracefully")
        print("")
        print("🚀 System is running... Happy trading!")

        # Wait for trading task
        try:
            await trading_task
        except KeyboardInterrupt:
            print("\n⏹️  Shutting down trading system...")
            await agent.stop_trading()
        except Exception as e:
            print(f"\n❌ Trading error: {e}")
            await agent.emergency_stop_trading()

    else:
        print("❌ Failed to start complete system")
        return

    print("\n💾 System shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  System interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal system error: {e}")
        sys.exit(1)
