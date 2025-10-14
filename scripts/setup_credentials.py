#!/usr/bin/env python3
"""
Setup script for configuring API credentials securely.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_trader.security.secrets import setup_aster_credentials, setup_news_credentials, setup_ai_credentials


def main():
    """Setup/update API credentials."""
    print("🔐 Rari Trade AI - Update Aster API Credentials")
    print("=" * 55)
    print("⚠️  Updating deprecated API keys with new ones")
    print()

    # Use no encryption for simplicity
    master_password = None
    print("⚠️  Running without encryption (not recommended for production)")

    print("\n📊 Aster DEX API Configuration")
    print("-" * 30)

    # Get new Aster API credentials
    aster_api_key = os.getenv('ASTER_API_KEY')
    if not aster_api_key:
        aster_api_key = input("Enter your new Aster API key: ").strip()

    aster_secret_key = os.getenv('ASTER_SECRET_KEY')
    if not aster_secret_key:
        aster_secret_key = input("Enter your new Aster secret key: ").strip()

    if not aster_api_key or not aster_secret_key:
        print("❌ Aster API key and secret are required")
        return

    print(f"✅ New API key: {aster_api_key[:20]}...")
    print(f"✅ New secret key: {aster_secret_key[:20]}...")
    print()

    # Setup Aster credentials
    setup_aster_credentials(aster_api_key, aster_secret_key, master_password)
    
    # Skip News and AI API setup for MVP focus on Aster trading
    print("Skipping News API configuration (optional for MVP).")
    print("Skipping AI Model API configuration (optional for MVP).")

    print("\n✅ Credential setup completed!")
    print("Aster DEX API credentials configured and saved.")

    # Auto-save to file
    from mcp_trader.security.secrets import get_secret_manager
    sm = get_secret_manager(master_password)
    if sm.save_secrets_to_file():
        print("✅ Credentials saved to .secrets.json")
    else:
        print("❌ Failed to save credentials to file")


if __name__ == "__main__":
    main()
