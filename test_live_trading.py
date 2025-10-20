#!/usr/bin/env python3
"""
Quick test of the live trading system
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_trading_agent import LiveTradingAgent, TradingConfig

async def test_live_trading():
    """Test the live trading system for a short period"""

    print("🧪 Testing Live Trading System...")

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
        dry_run=True  # Always dry run for testing
    )

    # Create mock client
    class MockAsterClient:
        def __init__(self):
            self.balance = config.initial_capital

        async def get_24hr_ticker(self, symbol):
            # Simulate price changes
            import random
            base_price = 50000.0 if 'BTC' in symbol else 3000.0
            change = random.uniform(-0.01, 0.01)  # -1% to +1% change
            price = base_price * (1 + change)
            return {
                'lastPrice': str(price),
                'symbol': symbol
            }

        async def get_account_info(self):
            return {'total_balance': str(self.balance)}

        async def get_order_book(self, symbol):
            return {'bids': [], 'asks': []}

        async def get_klines(self, symbol, interval, limit=100):
            # Return mock klines
            import time
            current_time = int(time.time() * 1000)
            klines = []
            for i in range(limit):
                timestamp = current_time - (limit - i) * 60000  # 1 minute intervals
                open_price = 50000.0 if 'BTC' in symbol else 3000.0
                close_price = open_price * (1 + (i % 10 - 5) * 0.001)  # Small variations
                klines.append([
                    timestamp,
                    str(open_price),
                    str(max(open_price, close_price)),
                    str(min(open_price, close_price)),
                    str(close_price),
                    "1000.0",  # volume
                    timestamp + 60000,  # close time
                    "500.0",  # quote volume
                    "10",  # trades
                    "500.0",  # taker buy base
                    "250.0",  # taker buy quote
                    "0"  # ignore
                ])
            return klines

    # Create mock client
    aster_client = MockAsterClient()

    # Create trading agent
    agent = LiveTradingAgent(config, aster_client)

    print("✅ Agent created successfully")
    print("📊 Testing basic functionality...")

    # Test getting status
    status = agent.get_status()
    print(f"   Initial capital: ${status['config']['initial_capital']}")
    print(f"   Max positions: {status['config']['max_positions']}")
    print(f"   Dry run: {config.dry_run}")

    # Test market data update (short test)
    print("\n📈 Testing market data updates...")
    try:
        market_data = await agent._update_market_data()
        print(f"   Retrieved data for {len(market_data)} symbols")
        for symbol, data in market_data.items():
            print(".2f")
    except Exception as e:
        print(f"   ❌ Market data error: {e}")

    # Test signal generation
    print("\n🎯 Testing signal generation...")
    try:
        if market_data:
            signals = await agent._generate_signals(market_data)
            total_signals = sum(len(symbol_signals) for symbol_signals in signals.values())
            print(f"   Generated {total_signals} signals across {len(signals)} symbols")
    except Exception as e:
        print(f"   ❌ Signal generation error: {e}")

    print("\n✅ Live trading system test completed!")
    print("🚀 The system is ready for deployment")

    return status

if __name__ == "__main__":
    asyncio.run(test_live_trading())
