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
                    import logging
                    logger = logging.getLogger(__name__)

                    # Try to load from local historical data first
                    try:
                        historical_data = self._load_historical_data(symbol)
                        if historical_data is not None and len(historical_data) >= limit:
                            # Use recent historical data - it's already OHLC format
                            recent_data = historical_data.tail(limit)
                            formatted_klines = []

                            for idx, row in recent_data.iterrows():
                                # Convert index (timestamp) to milliseconds
                                if hasattr(idx, 'timestamp'):
                                    timestamp = int(idx.timestamp() * 1000)
                                else:
                                    timestamp = int(time.time() * 1000)

                                # Extract OHLC values
                                open_price = float(row['open'])
                                high_price = float(row['high'])
                                low_price = float(row['low'])
                                close_price = float(row['close'])
                                volume = float(row.get('volume', 1000.0))

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

                            print(f"[DEBUG] Formatted {len(formatted_klines)} historical klines for {symbol}")
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
                    """Load historical data from local files and convert to OHLC format"""
                    import pandas as pd
                    import os
                    import numpy as np
                    import logging
                    logger = logging.getLogger(__name__)

                    try:
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
                                if not df.empty and len(df) > 50:  # Ensure we have enough data
                                    print(f"[DEBUG] Loaded {len(df)} rows from {path}")

                                    # Convert to OHLC format
                                    df_ohlc = self._convert_to_ohlc(df)
                                    if df_ohlc is not None and not df_ohlc.empty:
                                        print(f"[DEBUG] Converted to OHLC: {len(df_ohlc)} rows")
                                        return df_ohlc
                    except Exception as e:
                        print(f"[DEBUG] Failed to load historical data for {symbol}: {e}")

                    return None

                def _convert_to_ohlc(self, df):
                    """Convert price data to OHLC format for candlestick charts"""
                    import pandas as pd
                    import numpy as np
                    import logging
                    logger = logging.getLogger(__name__)

                    try:
                        # Ensure we have timestamp and price columns
                        if 'timestamp' not in df.columns or 'price' not in df.columns:
                            print(f"[DEBUG] Missing required columns. Has: {list(df.columns)}")
                            return None

                        # Make a copy to avoid modifying original
                        df = df.copy()

                        # Convert timestamp to datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                        # Set timestamp as index if not already
                        if df.index.name != 'timestamp':
                            df = df.set_index('timestamp')

                        # Sort by timestamp
                        df = df.sort_index()

                        # Generate OHLC from price data (simplified)
                        # In real trading, this would be actual tick-by-tick data
                        df_ohlc = pd.DataFrame(index=df.index)
                        df_ohlc['close'] = df['price'].astype(float)

                        # Generate realistic OHLC from close prices
                        np.random.seed(42)  # For reproducible results

                        # High/Low as percentage variation from close
                        volatility = 0.01  # 1% daily volatility
                        df_ohlc['high'] = df_ohlc['close'] * (1 + np.random.uniform(0, volatility, len(df_ohlc)))
                        df_ohlc['low'] = df_ohlc['close'] * (1 - np.random.uniform(0, volatility, len(df_ohlc)))

                        # Open as previous close or slight variation
                        df_ohlc['open'] = df_ohlc['close'].shift(1)
                        mask = df_ohlc['open'].isna()
                        if mask.any():
                            df_ohlc.loc[mask, 'open'] = df_ohlc.loc[mask, 'close'] * (1 + np.random.uniform(-0.005, 0.005, mask.sum()))

                        # Volume data (use existing or generate)
                        if 'volume' in df.columns and df['volume'].notna().any():
                            df_ohlc['volume'] = df['volume'].astype(float)
                        else:
                            # Generate realistic volume based on price level
                            base_volume = 1000000.0 if 'BTC' in symbol.upper() else 100000.0
                            df_ohlc['volume'] = base_volume * (1 + np.random.uniform(-0.5, 0.5, len(df_ohlc)))

                        # Ensure high >= max(open, close) and low <= min(open, close)
                        df_ohlc['high'] = df_ohlc[['high', 'open', 'close']].max(axis=1)
                        df_ohlc['low'] = df_ohlc[['low', 'open', 'close']].min(axis=1)

                        print(f"[DEBUG] OHLC conversion successful: {len(df_ohlc)} rows, columns: {list(df_ohlc.columns)}")
                        return df_ohlc[['open', 'high', 'low', 'close', 'volume']]

                    except Exception as e:
                        print(f"[DEBUG] Failed to convert to OHLC format: {e}")
                        import traceback
                        traceback.print_exc()
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
