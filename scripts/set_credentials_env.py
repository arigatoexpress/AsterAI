#!/usr/bin/env python3
"""
Set API credentials from environment variables
"""

import os
import json

def set_from_env():
    """Set API credentials from environment variables"""
    print("🔐 Setting API Credentials from Environment")
    print("="*50)
    
    # Check for environment variables
    api_key = os.environ.get('ASTER_API_KEY', '')
    api_secret = os.environ.get('ASTER_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("\n❌ Environment variables not found!")
        print("\nTo set them, run these commands in your terminal:")
        print("\nFor Windows PowerShell:")
        print('  $env:ASTER_API_KEY="your_api_key_here"')
        print('  $env:ASTER_API_SECRET="your_api_secret_here"')
        print("\nFor Windows CMD:")
        print('  set ASTER_API_KEY=your_api_key_here')
        print('  set ASTER_API_SECRET=your_api_secret_here')
        print("\nFor Linux/Mac:")
        print('  export ASTER_API_KEY="your_api_key_here"')
        print('  export ASTER_API_SECRET="your_api_secret_here"')
        print("\nThen run this script again.")
        return False
    
    print(f"✅ Found API Key (starts with: {api_key[:8]}...)")
    print(f"✅ Found API Secret (hidden)")
    
    # Load existing keys
    try:
        with open('.api_keys.json', 'r') as f:
            keys = json.load(f)
    except:
        keys = {}
    
    # Update with environment variables
    keys['aster_api_key'] = api_key
    keys['aster_secret_key'] = api_secret
    
    # Keep other keys
    keys.setdefault('alpha_vantage_key', 'WEO6JTK3E9WFRGRE')
    keys.setdefault('finnhub_key', 'd3ndn01r01qo7510l2c0d3ndn01r01qo7510l2cg')
    keys.setdefault('fred_api_key', 'a5b90245298d19b19abb6777beea54e1')
    keys.setdefault('newsapi_key', 'd725036479da4a4185537696e40b04f1')
    keys.setdefault('metals_api_key', 'Not set')
    
    # Save to file in local directory
    with open('local/.api_keys.json', 'w') as f:
        json.dump(keys, f, indent=2)
    
    print("\n✅ API credentials saved to .api_keys.json")
    print("\nNow you can run:")
    print("  python scripts/quick_api_test.py")
    print("  python scripts/test_and_debug_system.py")
    
    return True

if __name__ == "__main__":
    set_from_env()
