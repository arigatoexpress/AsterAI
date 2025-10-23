#!/usr/bin/env python3
"""
Test the Aggressive Perpetual Strategy
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.aggressive_perps_strategy import AggressivePerpsStrategy


async def test_aggressive_strategy():
    """Test the aggressive perpetual strategy."""
    print("🚀 Testing Aggressive Perpetual Strategy")
    print("=" * 50)

    # Initialize strategy
    strategy = AggressivePerpsStrategy(max_positions=5, max_position_size=0.10)

    # Test market data (EXTREME bullish trend with maximum momentum)
    bullish_data = {
        'prices': [50000, 50500, 51000, 52000, 53500, 55500, 58000, 61000, 64500, 68500,
                  73000, 78000, 83500, 89500, 96000, 103000, 110500, 118500, 127000, 136000],
        'volumes': [10000, 15000, 25000, 40000, 65000, 100000, 150000, 220000, 300000, 400000,
                   550000, 750000, 1000000, 1300000, 1700000, 2200000, 2800000, 3500000, 4300000, 5200000],
        'funding_rate': 0.0002  # Slightly positive funding (good for longs)
    }

    # Update market data cache first
    strategy.update_market_data('BTCUSDT', bullish_data)

    # Analyze market condition
    condition = strategy.analyze_market_condition('BTCUSDT', bullish_data)
    print(f"📊 Market Analysis: {condition.trend} trend, {condition.volatility} volatility")
    print(f"   Momentum: {condition.momentum:.3f}, Volume Score: {condition.volume_score:.3f}")

    # Test signal strength
    signal_strength = strategy.calculate_entry_signal_strength('BTCUSDT', bullish_data, condition)
    print(f"🎯 Signal Strength: {signal_strength:.3f}")

    # Test entry decision
    print(f"   Signal threshold: 0.6 (testing with extreme data)")
    entry_signal = strategy.should_enter_position('BTCUSDT', bullish_data)
    if entry_signal:
        print(f"🚀 ENTRY SIGNAL GENERATED:")
        print(f"   Symbol: {entry_signal['symbol']}")
        print(f"   Side: {entry_signal['side']}")
        print(f"   Leverage: {entry_signal['leverage']}x")
        print(f"   Position Size: {entry_signal['position_size_pct']:.1%}")
        print(f"   Signal Strength: {entry_signal['signal_strength']:.3f}")

        # Execute entry
        success = strategy.execute_entry(entry_signal)
        print(f"✅ Entry Execution: {'SUCCESS' if success else 'FAILED'}")

        # Check portfolio
        portfolio = strategy.get_portfolio_status()
        print(f"📊 Portfolio Status: {portfolio['active_positions']}/{portfolio['max_positions']} positions")

        # Test position management
        print("\n🔄 Testing Position Management...")

        # Test with profitable price movement (10% profit)
        current_price = bullish_data['prices'][-1] * 1.10
        exit_reason = strategy.should_exit_position(
            list(strategy.positions.values())[0], current_price, bullish_data
        )
        if exit_reason:
            print(f"💰 EXIT SIGNAL: {exit_reason.upper()} at ${current_price:.0f}")
            strategy.execute_exit('BTCUSDT', exit_reason)
        else:
            print("⏸️  No exit signal (holding position)")

    else:
        print("⏸️  No entry signal generated - signal strength too low")
        print("💡 This is normal - strategy requires strong confluence of factors")

    # Test signal analysis endpoint
    print("\n🎯 Testing Signal Analysis...")
    signal_summary = strategy.get_signal_summary('BTCUSDT')
    print(f"Signal Summary: {signal_summary}")

    print("\n✅ Aggressive Strategy Test Complete!")
    print("🎯 Strategy configured for maximum profit potential with calculated risk")


if __name__ == "__main__":
    asyncio.run(test_aggressive_strategy())
