#!/usr/bin/env python3
"""
Detailed API debugging script to identify exact issues
"""

import asyncio
import json
import sys
import os
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def debug_api_detailed():
    """Detailed debugging of Aster API connection"""
    print("🔍 Detailed Aster API Debugging")
    print("="*60)

    # Load credentials
    try:
        api_keys_path = 'local/.api_keys.json'
        if not os.path.exists(api_keys_path):
            api_keys_path = '.api_keys.json'

        with open(api_keys_path, 'r') as f:
            keys = json.load(f)

        api_key = keys.get('aster_api_key', '')
        api_secret = keys.get('aster_secret_key', '')

        if not api_key or api_key == '\x16' or len(api_key) < 10:
            print("❌ API credentials not properly configured")
            print("   Please run: python scripts/manual_update_keys.py")
            return False

        print(f"✅ API credentials loaded (key: {api_key[:12]}...)")

        # Set environment variables
        os.environ['ASTER_API_KEY'] = api_key
        os.environ['ASTER_API_SECRET'] = api_secret

    except Exception as e:
        print(f"❌ Error loading API credentials: {e}")
        return False

    # Import and initialize client
    try:
        from mcp_trader.execution.aster_client import AsterClient
        print("✅ AsterClient imported successfully")

        client = AsterClient(api_key, api_secret)
        print("✅ AsterClient initialized")

        # Use client as async context manager
        async with client:
            print("✅ AsterClient connected")

            # Test 1: Basic connectivity
            print("\n📡 Testing basic connectivity...")
            try:
                # Check if we can make HTTP requests at all
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    # Test a basic HTTP request to see if network works
                    try:
                        async with session.get('https://httpbin.org/get', timeout=10) as resp:
                            print(f"   ✅ Basic HTTP request works (status: {resp.status})")
                    except Exception as e:
                        print(f"   ❌ Basic HTTP request failed: {e}")
                        return False

            except Exception as e:
                print(f"❌ HTTP connectivity test failed: {e}")
                return False

            # Test 2: Aster API endpoints
            print("\n🔗 Testing Aster API endpoints...")

            # Test different endpoints to see what's working
            endpoints_to_test = [
                ('server_time', 'server_time'),
                ('ticker_24hr', 'get_24hr_ticker'),
                ('order_book', 'get_order_book'),
                ('account', 'get_account'),
                ('connectivity', 'test_connectivity')
            ]

            for endpoint_name, method_name in endpoints_to_test:
                print(f"\n   Testing {endpoint_name}...")

                try:
                    if method_name == 'server_time':
                        result = await client.get_server_time()
                    elif method_name == 'get_24hr_ticker':
                        result = await client.get_24hr_ticker('BTCUSDT')
                    elif method_name == 'get_order_book':
                        result = await client.get_order_book('BTCUSDT', limit=5)
                    elif method_name == 'get_account':
                        result = await client.get_account_info()
                    elif method_name == 'test_connectivity':
                        result = await client.test_connectivity()

                    if result is not None:
                        print(f"     ✅ {endpoint_name}: Success")
                        if isinstance(result, dict):
                            print(f"     📊 Response keys: {list(result.keys())[:3]}...")  # Show first 3 keys
                        elif isinstance(result, (list, tuple)):
                            print(f"     📊 Response length: {len(result)}")
                        else:
                            print(f"     📊 Response: {result}")
                    else:
                        print(f"     ⚠️  {endpoint_name}: Returned None")
                except Exception as e:
                    print(f"     ❌ {endpoint_name}: Failed - {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Test 3: Authentication status
            print("\n🔐 Testing authentication...")
            try:
                account_info = await client.get_account_info()
                if account_info and hasattr(account_info, 'balances'):
                    print("   ✅ Authentication successful")
                    print(f"   📊 Account has {len(account_info.balances)} balances")
                else:
                    print("   ⚠️  Authentication may need trading permissions")
                    print("   💡 Try using read-only API keys for testing")
            except Exception as e:
                print(f"   ❌ Authentication failed: {str(e)}")

            # Test 4: Market data
            print("\n📈 Testing market data...")
            symbols_to_test = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

            for symbol in symbols_to_test:
                print(f"   Testing {symbol}...")

                try:
                    ticker = await client.get_24hr_ticker(symbol)
                    if ticker and isinstance(ticker, dict):
                        price = ticker.get('lastPrice', 'N/A')
                        volume = ticker.get('volume', 'N/A')
                        print(f"     ✅ {symbol}: ${price} ({volume} volume)")
                    else:
                        print(f"     ❌ {symbol}: Invalid response format")
                except Exception as e:
                    print(f"     ❌ {symbol}: Failed - {str(e)}")

            print("\n" + "="*60)
            print("🔧 TROUBLESHOOTING TIPS:")
            print("="*60)
            print("1. 📋 Verify API credentials are correct")
            print("2. 🔐 Ensure API key has proper permissions")
            print("3. 🌐 Check network connectivity")
            print("4. ⏱️  Try again - API might be rate limited")
            print("5. 📖 Check Aster API documentation for correct endpoints")

            return True

    except Exception as e:
        print(f"❌ Error initializing AsterClient: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_api_detailed())
    print(f"\n{'✅' if result else '❌'} Debugging completed")
