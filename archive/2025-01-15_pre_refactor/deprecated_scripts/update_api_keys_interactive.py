#!/usr/bin/env python3
"""
Interactive script to update Aster API keys.
Prompts for keys if not found in environment variables.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mcp_trader.security.secrets import setup_aster_credentials, get_secret_manager

def main():
    print("🔑 Rari Trade AI - Update Aster API Keys (Interactive)")
    print("=" * 55)
    print("⚠️  Your old API keys are deprecated. Please update with new ones.")
    print()

    # Get API keys from environment variables or prompt
    api_key = os.getenv('ASTER_API_KEY')
    if not api_key:
        api_key = input("Enter your new Aster API key: ").strip()

    api_secret = os.getenv('ASTER_SECRET_KEY')
    if not api_secret:
        api_secret = input("Enter your new Aster secret key: ").strip()

    if not api_key or not api_secret:
        print("❌ Both API key and secret are required!")
        return 1

    print(f"✅ API Key: {api_key[:20]}...")
    print(f"✅ Secret Key: {api_secret[:20]}...")
    print()

    try:
        # Setup new credentials (no encryption for simplicity)
        setup_aster_credentials(api_key, api_secret, master_password=None)

        # Save to file
        sm = get_secret_manager()
        if sm.save_secrets_to_file():
            print("✅ Credentials updated and saved successfully!")
            print("📁 File: .secrets.json")
        else:
            print("❌ Failed to save credentials to file")
            return 1

        # Verify the update worked
        updated_key = sm.get_secret('ASTER_API_KEY')
        updated_secret = sm.get_secret('ASTER_SECRET_KEY')

        if updated_key == api_key and updated_secret == api_secret:
            print("✅ Verification successful - keys are properly stored")
            print()
            print("🚀 You can now run your trading platform with the new keys!")
            print("   Dashboard: python run_dashboard.py")
            print("   Live trading: python -m mcp_trader.agent.live_agent")
        else:
            print("❌ Verification failed - keys may not be properly stored")
            return 1

    except Exception as e:
        print(f"❌ Error updating credentials: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
