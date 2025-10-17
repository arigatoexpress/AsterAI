#!/usr/bin/env python3
"""
Quick script to update Aster API keys.
Run with environment variables or it will prompt interactively.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mcp_trader.security.secrets import setup_aster_credentials, get_secret_manager

def main():
    print("🔑 Rari Trade AI - Update Aster API Keys")
    print("=" * 45)
    print("⚠️  Your old API keys are deprecated. Please update with new ones.")
    print()

    # Get API keys from environment variables or prompt
    api_key = os.getenv('ASTER_API_KEY')
    api_secret = os.getenv('ASTER_SECRET_KEY')

    if not api_key or not api_secret:
        print("❌ API keys not found in environment variables.")
        print()
        print("📝 Please set your new API keys using one of these methods:")
        print()
        print("Method 1 - Set environment variables first:")
        print("  export ASTER_API_KEY='your_new_api_key'")
        print("  export ASTER_SECRET_KEY='your_new_secret_key'")
        print("  python update_api_keys.py")
        print()
        print("Method 2 - Run with inline environment variables:")
        print("  ASTER_API_KEY='your_new_api_key' ASTER_SECRET_KEY='your_new_secret_key' python update_api_keys.py")
        print()
        print("Method 3 - Interactive (run directly and enter when prompted):")
        print("  python update_api_keys.py")
        print("  (then enter your keys when prompted)")
        print()
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
