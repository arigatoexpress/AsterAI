#!/usr/bin/env python3
"""
Test Aster DEX authenticated API endpoints manually.
This script tests if the API credentials are working correctly.
"""

import os
import sys
import time
import hmac
import hashlib
import requests
from typing import Dict, Optional
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_credentials_from_gcp() -> tuple[str, str]:
    """Get credentials from Google Secrets Manager"""
    try:
        from google.cloud import secretmanager_v1
        client = secretmanager_v1.SecretManagerServiceClient()
        project_id = "quant-ai-trader-credits"

        # Get API key
        api_key_response = client.access_secret_version(
            request={"name": f"projects/{project_id}/secrets/ASTER_API_KEY/versions/latest"}
        )
        api_key = api_key_response.payload.data.decode('UTF-8')

        # Get secret key
        secret_key_response = client.access_secret_version(
            request={"name": f"projects/{project_id}/secrets/ASTER_SECRET_KEY/versions/latest"}
        )
        secret_key = secret_key_response.payload.data.decode('UTF-8')

        return api_key, secret_key
    except Exception as e:
        print(f"❌ Failed to get credentials from GCP: {e}")
        return None, None

def generate_signature(secret_key: str, method: str, endpoint: str, params: Dict, timestamp: str, recv_window: str = "5000") -> str:
    """Generate HMAC SHA256 signature for Aster API (matching AsterClient implementation)"""
    # Create all params including timestamp and recvWindow
    all_params = params.copy()
    all_params['timestamp'] = timestamp
    all_params['recvWindow'] = recv_window

    # Sort parameters by key and create query string
    query_string = "&".join([f"{k}={v}" for k, v in sorted(all_params.items())])

    # Create the message to sign (method + endpoint + query_string)
    message = f"{method}{endpoint}{query_string}"

    # Generate signature using HMAC SHA256
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return signature

def test_account_info(api_key: str, secret_key: str) -> Dict:
    """Test account info endpoint"""
    print("\n🔍 Testing Account Info Endpoint...")

    base_url = "https://fapi.asterdex.com"
    endpoint = "/fapi/v4/account"
    timestamp = str(int(time.time() * 1000))

    # Only include additional params, timestamp and recvWindow added by signature function
    params = {}

    signature = generate_signature(secret_key, "GET", endpoint, params, timestamp)
    params["signature"] = signature
    params["timestamp"] = timestamp
    params["recvWindow"] = "5000"

    headers = {
        "X-MBX-APIKEY": api_key
    }

    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Account Info Retrieved Successfully!")
            print(f"Total Balance: {data.get('totalWalletBalance', 'N/A')}")
            print(f"Available Balance: {data.get('availableBalance', 'N/A')}")
            return {"success": True, "data": data}
        else:
            print(f"❌ API Error: {response.text}")
            return {"success": False, "error": response.text, "status_code": response.status_code}

    except Exception as e:
        print(f"❌ Request Failed: {e}")
        return {"success": False, "error": str(e)}

def test_account_balance(api_key: str, secret_key: str) -> Dict:
    """Test account balance endpoint"""
    print("\n🔍 Testing Account Balance Endpoint...")

    base_url = "https://fapi.asterdex.com"
    endpoint = "/fapi/v2/balance"
    timestamp = str(int(time.time() * 1000))

    # Only include additional params, timestamp and recvWindow added by signature function
    params = {}

    signature = generate_signature(secret_key, "GET", endpoint, params, timestamp)
    params["signature"] = signature
    params["timestamp"] = timestamp
    params["recvWindow"] = "5000"

    headers = {
        "X-MBX-APIKEY": api_key
    }

    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Account Balance Retrieved Successfully!")
            # Show first few balances
            for balance in data[:3]:
                if float(balance.get('balance', 0)) > 0:
                    print(f"  {balance.get('asset')}: {balance.get('balance')} (Free: {balance.get('availableBalance')})")
            return {"success": True, "data": data}
        else:
            print(f"❌ API Error: {response.text}")
            return {"success": False, "error": response.text, "status_code": response.status_code}

    except Exception as e:
        print(f"❌ Request Failed: {e}")
        return {"success": False, "error": str(e)}

def test_positions(api_key: str, secret_key: str) -> Dict:
    """Test positions endpoint"""
    print("\n🔍 Testing Positions Endpoint...")

    base_url = "https://fapi.asterdex.com"
    endpoint = "/fapi/v2/positionRisk"
    timestamp = str(int(time.time() * 1000))

    # Only include additional params, timestamp and recvWindow added by signature function
    params = {}

    signature = generate_signature(secret_key, "GET", endpoint, params, timestamp)
    params["signature"] = signature
    params["timestamp"] = timestamp
    params["recvWindow"] = "5000"

    headers = {
        "X-MBX-APIKEY": api_key
    }

    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Positions Retrieved Successfully!")
            # Show positions with non-zero amounts
            open_positions = [pos for pos in data if float(pos.get('positionAmt', 0)) != 0]
            if open_positions:
                for pos in open_positions[:3]:
                    print(f"  {pos.get('symbol')}: {pos.get('positionAmt')} @ {pos.get('entryPrice')}")
            else:
                print("  No open positions")
            return {"success": True, "data": data}
        else:
            print(f"❌ API Error: {response.text}")
            return {"success": False, "error": response.text, "status_code": response.status_code}

    except Exception as e:
        print(f"❌ Request Failed: {e}")
        return {"success": False, "error": str(e)}

def test_binance_api(api_key: str, secret_key: str) -> bool:
    """Test if credentials work with Binance API (to check if they're Binance credentials)"""
    print("\n🔍 Testing if credentials work with Binance API...")

    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v2/balance"
    timestamp = str(int(time.time() * 1000))

    params = {}
    signature = generate_signature(secret_key, "GET", endpoint, params, timestamp)
    params["signature"] = signature
    params["timestamp"] = timestamp
    params["recvWindow"] = "5000"

    headers = {
        "X-MBX-APIKEY": api_key
    }

    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Credentials work with Binance API!")
            return True
        elif "invalid" in response.text.lower():
            print("❌ Credentials do not work with Binance API either")
            return False
        else:
            print(f"Binance API response: {response.status_code} - {response.text[:100]}")
            return False
    except Exception as e:
        print(f"❌ Binance API test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Aster DEX API Authentication Test")
    print("=" * 50)

    # Get credentials from GCP
    api_key, secret_key = get_credentials_from_gcp()

    if not api_key or not secret_key:
        print("❌ Could not retrieve API credentials from Google Secrets Manager")
        print("Please ensure ASTER_API_KEY and ASTER_SECRET_KEY secrets exist and are accessible")
        return

    print(f"✅ Retrieved API credentials (Key: {api_key[:8]}..., Secret: {secret_key[:8]}...)")

    # First test if these are actually Binance credentials
    if test_binance_api(api_key, secret_key):
        print("\n⚠️  WARNING: These appear to be Binance API credentials, not Aster DEX!")
        print("   Please update Google Secrets Manager with Aster DEX credentials.")
        return

    # Test different endpoints
    results = {}

    # Test account info
    results['account_info'] = test_account_info(api_key, secret_key)

    # Test account balance
    results['account_balance'] = test_account_balance(api_key, secret_key)

    # Test positions
    results['positions'] = test_positions(api_key, secret_key)

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    successful_tests = sum(1 for result in results.values() if result.get('success', False))
    total_tests = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {successful_tests}/{total_tests} tests passed")

    if successful_tests == total_tests:
        print("🎉 All API authentication tests passed! Credentials are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check:")
        print("   1. API credentials are valid for Aster DEX")
        print("   2. Futures trading is enabled on the account")
        print("   3. API key has necessary permissions")
        print("   4. You're using mainnet credentials (not testnet)")

if __name__ == "__main__":
    main()
