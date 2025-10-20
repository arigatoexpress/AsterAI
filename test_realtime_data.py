#!/usr/bin/env python3
"""
Test script to verify real-time market data is working
"""

import asyncio
import sys
sys.path.insert(0, '.')

from realtime_price_fetcher import RealTimePriceFetcher

async def test_realtime_data():
    """Test real-time data integration"""
    print("🧪 Testing Real-Time Data Integration...")

    async with RealTimePriceFetcher() as fetcher:
        print("📈 Fetching real-time market data...")

        # Test current prices
        prices = await fetcher.get_current_prices(['BTC', 'ETH', 'SOL', 'ADA'])

        if prices:
            print("✅ Real-time prices fetched successfully:")
            print("-" * 50)
            for symbol, data in prices.items():
                price = data.get('price', 0)
                change = data.get('change_24h', 0)
                source = data.get('source', 'unknown')
                timestamp = data.get('timestamp', 'unknown')

                print("6s")

            print("-" * 50)

            # Test caching
            print("📊 Testing cache functionality...")
            cached = fetcher.get_cached_prices()
            print(f"   Cached prices: {len(cached)} symbols")

            # Wait and test again
            print("⏳ Waiting 5 seconds and fetching again...")
            await asyncio.sleep(5)
            prices2 = await fetcher.get_current_prices(['BTC'])

            if prices2:
                print("✅ Second fetch successful - APIs are working!")
                return True
            else:
                print("❌ Second fetch failed")
                return False

        else:
            print("❌ Failed to fetch real-time prices")
            print("   Possible issues:")
            print("   • No internet connection")
            print("   • API rate limits")
            print("   • API endpoints blocked")
            print("   • All APIs returning errors")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_realtime_data())
    if success:
        print("\n🎉 Real-time data pipelines are functional!")
        print("   Your dashboard should now show current market prices.")
    else:
        print("\n❌ Real-time data pipelines need troubleshooting.")
        print("   Check internet connection and API availability.")
