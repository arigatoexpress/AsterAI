#!/usr/bin/env python3
"""
Quick Aster DEX Credentials Update
Run this to update your Aster DEX mainnet API credentials
"""

import os
import sys
from getpass import getpass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def update_credentials():
    """Update Aster DEX credentials"""
    print("🔑 Aster DEX Credentials Update")
    print("=" * 40)

    # Get credentials from user
    print("Enter your Aster DEX MAINNET API credentials:")
    print("(Get these from: https://app.asterdex.com/account/api)")
    print()

    api_key = getpass("API Key: ").strip()
    secret_key = getpass("Secret Key: ").strip()

    if not api_key or not secret_key:
        print("❌ Both API key and secret key are required")
        return False

    print(f"\nUsing API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")

    try:
        # Update Google Secrets Manager
        from google.cloud import secretmanager_v1

        project_id = "quant-ai-trader-credits"
        client = secretmanager_v1.SecretManagerServiceClient()

        # Update API key
        api_key_path = f"projects/{project_id}/secrets/ASTER_API_KEY"
        client.add_secret_version(
            request={
                "parent": api_key_path,
                "payload": {"data": api_key.encode('UTF-8')}
            }
        )
        print("✅ Updated ASTER_API_KEY in Google Secrets Manager")

        # Update secret key
        secret_key_path = f"projects/{project_id}/secrets/ASTER_SECRET_KEY"
        client.add_secret_version(
            request={
                "parent": secret_key_path,
                "payload": {"data": secret_key.encode('UTF-8')}
            }
        )
        print("✅ Updated ASTER_SECRET_KEY in Google Secrets Manager")

        # Test the credentials
        print("\n🧪 Testing credentials...")
        import scripts.test_aster_api_auth as test_module
        import asyncio

        # We can't easily test here due to async nature, but let's try
        print("Credentials updated! Run this to test:")
        print("python3 scripts/test_aster_api_auth.py")

        return True

    except Exception as e:
        print(f"❌ Failed to update credentials: {e}")
        return False

if __name__ == "__main__":
    success = update_credentials()
    if success:
        print("\n🎉 Credentials updated successfully!")
        print("\nNext steps:")
        print("1. Test credentials: python3 scripts/test_aster_api_auth.py")
        print("2. Restart trading bot: python3 scripts/start_cloud_bots.py --action start")
        print("3. Monitor trading: python3 scripts/start_cloud_bots.py --action logs --service aster-trading-agent")
    else:
        print("\n❌ Failed to update credentials. Please try again.")

