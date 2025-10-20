#!/usr/bin/env python3
"""
Quick API connection test
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_api():
    """Quick test of Aster API connection"""
    print("Quick Aster API Test")
    print("="*50)
    
    try:
        # Load credentials
        api_keys_path = 'local/.api_keys.json'
        if not os.path.exists(api_keys_path):
            api_keys_path = '.api_keys.json'

        with open(api_keys_path, 'r') as f:
            keys = json.load(f)
        
        api_key = keys.get('aster_api_key', '')
        api_secret = keys.get('aster_secret_key', '')
        
        if not api_key or api_key == '\x16' or len(api_key) < 10:
            print("❌ API credentials not found or invalid")
            print("Please run: python scripts/update_api_keys.py")
            return False
        
        print(f"✅ Found API credentials (key starts with: {api_key[:8]}...)")
        
        # Set environment variables
        os.environ['ASTER_API_KEY'] = api_key
        os.environ['ASTER_API_SECRET'] = api_secret
        
        # Import and test
        from mcp_trader.execution.aster_client import AsterClient

        print("\n🔌 Connecting to Aster API...")
        client = AsterClient(api_key, api_secret)

        # Use client as async context manager
        async with client:
            # Test 1: Get ticker
            print("\n📊 Testing market data...")
            try:
                ticker = await client.get_24hr_ticker('BTCUSDT')
                if ticker:
                    # Handle both dict and object responses
                    if hasattr(ticker, 'last_price'):
                        price = ticker.last_price
                        volume = ticker.volume
                    elif isinstance(ticker, dict):
                        price = ticker.get('lastPrice', 0)
                        volume = ticker.get('volume', 0)
                    else:
                        price = getattr(ticker, 'lastPrice', 'N/A')
                        volume = getattr(ticker, 'volume', 'N/A')

                    print(f"✅ BTC/USDT Price: ${price}")
                    print(f"✅ 24h Volume: ${volume:,.2f}")
                else:
                    print("⚠️ No ticker data returned")
            except Exception as e:
                print(f"❌ Market data error: {str(e)}")
                return False

            # Test 2: Get account (if available)
            print("\n👤 Testing account access...")
            try:
                account = await client.get_account_info()
                if account:
                    print("✅ Account connected")
                    # Handle different account info formats
                    if hasattr(account, 'balances') and account.balances:
                        balance = account.balances[0].free if account.balances else 0
                    else:
                        balance = getattr(account, 'total_balance', 0)
                    print(f"✅ Balance: {balance} USDC")
                else:
                    print("⚠️ No account data (may need trading permissions)")
            except Exception as e:
                print(f"⚠️ Account access not available: {str(e)}")
                print("   (This is normal if using read-only API keys)")
        
        print("\n🎉 API Connection Successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_api())
    sys.exit(0 if result else 1)