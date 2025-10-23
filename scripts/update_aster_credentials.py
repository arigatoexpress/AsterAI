#!/usr/bin/env python3
"""
Update Aster DEX Credentials - Local and Google Secrets Manager
"""

import os
import sys
import json
from pathlib import Path
from getpass import getpass
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_local_credentials(api_key: str, secret_key: str):
    """Update local credential files"""
    print("📝 Updating local credentials...")

    # Update .env file
    env_path = Path('.env')
    env_content = []

    # Read existing .env if it exists
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_content = f.readlines()

    # Remove existing Aster credentials
    env_content = [line for line in env_content if not line.startswith(('ASTER_API_KEY=', 'ASTER_SECRET_KEY='))]

    # Add new credentials
    env_content.extend([
        f"ASTER_API_KEY={api_key}\n",
        f"ASTER_SECRET_KEY={secret_key}\n"
    ])

    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(env_content)

    print(f"✅ Updated {env_path}")

    # Update .api_keys.json file
    api_keys_path = Path('.api_keys.json')
    api_keys_data = {}

    # Read existing file if it exists
    if api_keys_path.exists():
        try:
            with open(api_keys_path, 'r') as f:
                api_keys_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read existing {api_keys_path}: {e}")

    # Update Aster credentials
    api_keys_data['aster_api_key'] = api_key
    api_keys_data['aster_secret_key'] = secret_key

    # Write back to .api_keys.json
    with open(api_keys_path, 'w') as f:
        json.dump(api_keys_data, f, indent=2)

    print(f"✅ Updated {api_keys_path}")

    return True

def update_gcp_secrets(api_key: str, secret_key: str):
    """Update Google Secrets Manager"""
    print("☁️ Updating Google Secrets Manager...")

    try:
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
        print("✅ Updated ASTER_API_KEY in GCP Secret Manager")

        # Update secret key
        secret_key_path = f"projects/{project_id}/secrets/ASTER_SECRET_KEY"
        client.add_secret_version(
            request={
                "parent": secret_key_path,
                "payload": {"data": secret_key.encode('UTF-8')}
            }
        )
        print("✅ Updated ASTER_SECRET_KEY in GCP Secret Manager")

        return True

    except Exception as e:
        print(f"❌ Failed to update GCP secrets: {e}")
        return False

def test_api_credentials(api_key: str, secret_key: str):
    """Test the API credentials by making a test call"""
    print("🧪 Testing API credentials...")

    try:
        from mcp_trader.execution.aster_client import AsterClient

        # Create test client
        client = AsterClient(api_key=api_key, secret_key=secret_key)

        # Test connectivity (this will test authentication)
        async def test_connection():
            try:
                # Test server time (simple endpoint)
                server_time = await client.get_server_time()
                print(f"✅ Server time: {server_time}")

                # Test account info (requires authentication)
                account_info = await client.get_account_info()
                print(f"✅ Account balance: ${account_info.total_balance:.2f}")
                print(f"   Available: ${account_info.available_balance:.2f}")

                return True

            except Exception as e:
                print(f"❌ API test failed: {e}")
                return False

        # Run the async test
        import asyncio
        return asyncio.run(test_connection())

    except Exception as e:
        print(f"❌ Failed to create test client: {e}")
        return False

def main():
    """Main function"""
    print("🔐 Aster DEX Credentials Update Tool")
    print("=" * 50)
    print("This tool will:")
    print("1. Update local credential files (.env, .api_keys.json)")
    print("2. Push credentials to Google Secrets Manager")
    print("3. Test API connectivity")
    print()

    # Get credentials from user
    print("Enter your Aster DEX mainnet API credentials:")
    api_key = getpass("API Key: ").strip()
    secret_key = getpass("Secret Key: ").strip()

    if not api_key or not secret_key:
        print("❌ Both API key and secret key are required")
        return

    print(f"\nUsing API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")

    # Step 1: Update local credentials
    print("\n" + "="*50)
    print("STEP 1: Updating Local Credentials")
    print("="*50)

    if not update_local_credentials(api_key, secret_key):
        print("❌ Failed to update local credentials")
        return

    # Step 2: Update GCP secrets
    print("\n" + "="*50)
    print("STEP 2: Updating Google Secrets Manager")
    print("="*50)

    if not update_gcp_secrets(api_key, secret_key):
        print("⚠️ GCP update failed, but local credentials were updated")
        print("You can retry GCP update later or check your GCP authentication")

    # Step 3: Test credentials
    print("\n" + "="*50)
    print("STEP 3: Testing API Credentials")
    print("="*50)

    if test_api_credentials(api_key, secret_key):
        print("\n🎉 SUCCESS! Aster DEX credentials are working correctly")
        print("\n🚀 Your trading bot is now ready for live trading!")
        print("\nNext steps:")
        print("1. Check bot status: python3 scripts/start_cloud_bots.py --action status")
        print("2. Monitor trading: python3 scripts/start_cloud_bots.py --action logs --service aster-trading-agent")
        print("3. Access dashboard: https://aster-enhanced-dashboard-880429861698.us-central1.run.app")
    else:
        print("\n❌ API credentials test failed")
        print("Please verify:")
        print("- API credentials are for Aster DEX mainnet (not testnet)")
        print("- Futures trading is enabled on your account")
        print("- API key has necessary permissions")
        print("\nYou can still proceed with demo mode, but live trading will be limited")

if __name__ == "__main__":
    main()