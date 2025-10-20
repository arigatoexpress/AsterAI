#!/usr/bin/env python3
"""
Update API keys directly - enter your credentials when prompted
"""

import json
import os

def update_api_keys():
    """Update API keys with user input"""
    print("🔐 Updating Aster API Credentials")
    print("="*50)
    
    # Load existing keys
    api_keys_file = ".api_keys.json"
    
    try:
        with open(api_keys_file, 'r') as f:
            keys = json.load(f)
    except:
        keys = {}
    
    print("\nPlease enter your Aster API credentials:")
    print("(You can find these in your Aster account settings)")
    
    # Get credentials
    api_key = input("Aster API Key: ").strip()
    api_secret = input("Aster API Secret: ").strip()
    
    if not api_key or not api_secret:
        print("\n❌ Both API key and secret are required!")
        return False
    
    # Update the keys
    keys['aster_api_key'] = api_key
    keys['aster_secret_key'] = api_secret
    
    # Keep other keys
    keys.setdefault('alpha_vantage_key', 'WEO6JTK3E9WFRGRE')
    keys.setdefault('finnhub_key', 'd3ndn01r01qo7510l2c0d3ndn01r01qo7510l2cg')
    keys.setdefault('fred_api_key', 'a5b90245298d19b19abb6777beea54e1')
    keys.setdefault('newsapi_key', 'd725036479da4a4185537696e40b04f1')
    keys.setdefault('metals_api_key', 'Not set')
    
    # Save to file
    with open(api_keys_file, 'w') as f:
        json.dump(keys, f, indent=2)
    
    print("\n✅ API keys updated successfully!")
    
    # Also set environment variables
    os.environ['ASTER_API_KEY'] = api_key
    os.environ['ASTER_API_SECRET'] = api_secret
    
    # Verify
    print("\nVerifying saved credentials...")
    with open(api_keys_file, 'r') as f:
        saved_keys = json.load(f)
        if saved_keys.get('aster_api_key') == api_key:
            print("✅ Credentials verified and saved correctly!")
            return True
        else:
            print("❌ Error saving credentials")
            return False

if __name__ == "__main__":
    update_api_keys()
