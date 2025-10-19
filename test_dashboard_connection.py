#!/usr/bin/env python3
"""
Test script to verify dashboard can connect to the trading bot.
"""

import requests
import json

BOT_URL = "https://aster-trading-agent-880429861698.us-central1.run.app"

def test_connection():
    """Test all bot endpoints."""
    print("🔍 Testing Trading Bot Connection...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint:")
    try:
        response = requests.get(f"{BOT_URL}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data.get('status', 'unknown')}")
            print(f"   ✅ Running: {data.get('running', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test status endpoint
    print("\n2. Testing Status Endpoint:")
    try:
        response = requests.get(f"{BOT_URL}/status", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status', 'unknown')}")
            print(f"   ✅ Agent Type: {data.get('agent_type', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test debug endpoint
    print("\n3. Testing Debug Endpoint:")
    try:
        response = requests.get(f"{BOT_URL}/debug", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Environment Variables:")
            for key, value in data.get('environment_variables', {}).items():
                print(f"      {key}: {value}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test root endpoint
    print("\n4. Testing Root Endpoint:")
    try:
        response = requests.get(f"{BOT_URL}/", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Service: {data.get('service', 'unknown')}")
            print(f"   ✅ Version: {data.get('version', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Dashboard should now be able to connect to the bot!")
    print("📊 Open http://localhost:8081 in your browser to view the dashboard")

if __name__ == "__main__":
    test_connection()
