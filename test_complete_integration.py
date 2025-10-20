#!/usr/bin/env python3
"""
Complete Integration Test for Aster AI Trading System
Tests all components: data loading, API integration, trading signals
"""

import asyncio
import sys
import os
sys.path.insert(0, '.')

from start_complete_system import main as start_system
from realtime_price_fetcher import RealTimePriceFetcher

async def test_integration():
    """Test complete system integration"""
    print("="*80)
    print("🧪 ASTER AI COMPLETE SYSTEM INTEGRATION TEST")
    print("="*80)

    # Test 1: Real-time price fetching (Binance should work with Iceland VPN)
    print("\n📊 Test 1: Real-Time Price Fetching")
    print("-"*80)
    async with RealTimePriceFetcher() as fetcher:
        prices = await fetcher.get_current_prices(['BTC', 'ETH', 'SOL', 'ADA'])
        if prices:
            print("✅ Price fetching successful!")
            for symbol, data in prices.items():
                source = data.get('source', 'unknown')
                price = data.get('price', 0)
                print(f"   {symbol}: ${price:,.2f} (from {source})")

            # Check if Binance is working
            binance_working = any(data.get('source') == 'binance' for data in prices.values())
            if binance_working:
                print("✅ Binance API is working with Iceland VPN!")
            else:
                print("⚠️  Binance not responding, using fallback sources")
        else:
            print("❌ Price fetching failed")
            return False

    # Test 2: Historical data loading
    print("\n📂 Test 2: Historical Data Loading")
    print("-"*80)
    try:
        import pandas as pd

        btc_path = "data/historical/crypto/btc.parquet"
        eth_path = "data/historical/crypto/eth.parquet"

        if os.path.exists(btc_path):
            df_btc = pd.read_parquet(btc_path)
            print(f"✅ BTC historical data loaded: {len(df_btc)} rows")
            print(f"   Columns: {list(df_btc.columns)}")
            print(f"   Date range: {df_btc['timestamp'].min()} to {df_btc['timestamp'].max()}")
        else:
            print("❌ BTC historical data not found")

        if os.path.exists(eth_path):
            df_eth = pd.read_parquet(eth_path)
            print(f"✅ ETH historical data loaded: {len(df_eth)} rows")
            print(f"   Columns: {list(df_eth.columns)}")
            print(f"   Date range: {df_eth['timestamp'].min()} to {df_eth['timestamp'].max()}")
        else:
            print("❌ ETH historical data not found")

    except Exception as e:
        print(f"❌ Historical data loading failed: {e}")
        return False

    # Test 3: Data merging (historical + API)
    print("\n🔗 Test 3: Data Merging (Historical + API)")
    print("-"*80)
    try:
        # Load historical
        df_historical = pd.read_parquet("data/historical/crypto/btc.parquet")

        # Fetch current API data
        async with RealTimePriceFetcher() as fetcher:
            current_prices = await fetcher.get_current_prices(['BTC'])

        if current_prices and 'BTC' in current_prices:
            hist_latest = df_historical['price'].iloc[-1]
            api_current = current_prices['BTC']['price']

            print(f"✅ Data merging test:")
            print(f"   Historical latest: ${hist_latest:,.2f}")
            print(f"   API current: ${api_current:,.2f}")
            print(f"   Difference: ${abs(api_current - hist_latest):,.2f}")
            print(f"   Data is {'current' if abs(api_current - hist_latest) < 5000 else 'outdated'}")
        else:
            print("⚠️  API data not available for merge test")

    except Exception as e:
        print(f"❌ Data merging test failed: {e}")

    # Test 4: Trading signal generation
    print("\n🎯 Test 4: Trading Signal Generation")
    print("-"*80)
    try:
        from live_trading_agent import LiveTradingAgent, TradingConfig

        config = TradingConfig(
            initial_capital=100.0,
            dry_run=True,
            trading_pairs=["BTCUSDT", "ETHUSDT"]
        )

        # Create mock client with proper methods
        class TestMockClient:
            def __init__(self):
                self.balance = 100.0

            async def get_24hr_ticker(self, symbol):
                # Use real Binance prices
                async with RealTimePriceFetcher() as fetcher:
                    prices = await fetcher.get_current_prices(['BTC', 'ETH'])
                    base_symbol = symbol.replace('USDT', '')
                    if base_symbol in prices:
                        return {'lastPrice': str(prices[base_symbol]['price']), 'symbol': symbol}
                return {'lastPrice': '50000.0', 'symbol': symbol}

            async def get_account_info(self):
                class AccountInfo:
                    def __init__(self, balance):
                        self.total_balance = str(balance)
                return AccountInfo(self.balance)

            async def get_order_book(self, symbol):
                return {'bids': [], 'asks': []}

            async def get_klines(self, symbol, interval, limit=100):
                # Return simple but valid kline data
                current_time = int(asyncio.get_event_loop().time() * 1000)
                base_price = 110000.0 if 'BTC' in symbol else 2600.0

                klines = []
                for i in range(limit):
                    timestamp = current_time - (limit - i) * 60000
                    price = base_price * (1 + (i % 10 - 5) * 0.001)  # Small variation
                    klines.append([
                        timestamp,
                        str(price),  # open
                        str(price * 1.001),  # high
                        str(price * 0.999),  # low
                        str(price),  # close
                        "1000.0",  # volume
                        timestamp + 60000,
                        "500.0",
                        "10",
                        "500.0",
                        "250.0",
                        "0"
                    ])
                return klines

        agent = LiveTradingAgent(config, TestMockClient())
        print("✅ Trading agent created successfully")

        # Test market data update
        market_data = await agent._update_market_data()
        print(f"✅ Market data updated: {len(market_data)} symbols")

        # Test signal generation
        signals = await agent._generate_signals(market_data)
        total_signals = sum(len(s) for s in signals.values())
        print(f"✅ Trading signals generated: {total_signals} signals")

        return True

    except Exception as e:
        print(f"❌ Trading signal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("="*80)

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)

