#!/usr/bin/env python3
"""
Setup Aster API credentials securely
"""

import json
import os
import getpass

def setup_api_credentials():
    """Setup API credentials interactively"""
    print("🔐 AsterAI API Credential Setup")
    print("="*50)
    
    # Check if file exists
    api_keys_file = ".api_keys.json"
    
    # Load existing keys
    existing_keys = {}
    if os.path.exists(api_keys_file):
        with open(api_keys_file, 'r') as f:
            existing_keys = json.load(f)
    
    print("\nEnter your Aster API credentials (input is hidden for security):")
    
    # Get API key
    api_key = getpass.getpass("Aster API Key: ").strip()
    if not api_key:
        print("❌ API Key cannot be empty!")
        return False
        
    # Get API secret
    api_secret = getpass.getpass("Aster API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret cannot be empty!")
        return False
    
    # Update credentials
    existing_keys['aster_api_key'] = api_key
    existing_keys['aster_secret_key'] = api_secret
    
    # Save to file
    with open(api_keys_file, 'w') as f:
        json.dump(existing_keys, f, indent=2)
    
    print("\n✅ API credentials saved successfully!")
    
    # Set environment variables for current session
    os.environ['ASTER_API_KEY'] = api_key
    os.environ['ASTER_API_SECRET'] = api_secret
    
    print("✅ Environment variables set for current session")
    
    return True

if __name__ == "__main__":
    setup_api_credentials()
