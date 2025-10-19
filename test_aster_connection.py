#!/usr/bin/env python3
"""
Test script to check Aster DEX connection and API endpoints.
"""

import asyncio
import os
import sys
from mcp_trader.execution.aster_client import AsterClient

async def test_connection():
    """Test basic connection to Aster DEX."""
    api_key = os.getenv("ASTER_API_KEY")
    secret_key = os.getenv("ASTER_API_SECRET")
    
    if not api_key or not secret_key:
        print("❌ ASTER_API_KEY and ASTER_API_SECRET environment variables must be set")
        return False
    
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"🔑 Secret Key: {secret_key[:8]}...")
    
    try:
        client = AsterClient(api_key=api_key, secret_key=secret_key)
        
        # Test basic connectivity
        print("\n🔍 Testing basic connectivity...")
        try:
            ping_result = await client.ping()
            print(f"✅ Ping successful: {ping_result}")
        except Exception as e:
            print(f"❌ Ping failed: {e}")
        
        # Test server time
        print("\n🕐 Testing server time...")
        try:
            server_time = await client.get_server_time()
            print(f"✅ Server time: {server_time}")
        except Exception as e:
            print(f"❌ Server time failed: {e}")
        
        # Test ticker data
        print("\n📊 Testing ticker data...")
        try:
            ticker = await client.get_24hr_ticker("BTCUSDT")
            print(f"✅ Ticker data: {ticker}")
        except Exception as e:
            print(f"❌ Ticker data failed: {e}")
        
        # Test account info
        print("\n💰 Testing account info...")
        try:
            account_info = await client.get_account_info()
            print(f"✅ Account info: {account_info}")
        except Exception as e:
            print(f"❌ Account info failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection())
